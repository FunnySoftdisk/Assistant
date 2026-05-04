"""
Data Preprocessing Script for Qwen3 Fine-tuning
Loads data from ModelScope and converts to Qwen3 chat format
"""

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import traceback

from config.train_config import DATA_CONFIG, MODEL_CONFIG, TrainingConfig


@dataclass
class DataPreprocessor:
    """Handles data loading and preprocessing for Qwen3 fine-tuning"""

    dataset_name: str = DATA_CONFIG["dataset_name"]
    subset_name: str = DATA_CONFIG["subset_name"]
    split: str = DATA_CONFIG["split"]
    output_path: str = DATA_CONFIG["processed_data_path"]
    model_path: str = MODEL_CONFIG["model_path"]
    max_seq_length: int = 2048

    def __post_init__(self):
        self.tokenizer = None

    def load_tokenizer(self):
        """Load the tokenizer"""
        print(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_parallel(
            self.model_path,
            trust_remote_code=True,
            use_fast=False
        )
        # Set chat template if available
        if hasattr(self.tokenizer, 'chat_template'):
            print("Tokenizer has chat template")
        return self.tokenizer

    def load_raw_data(self) -> Dataset:
        """Load dataset from ModelScope"""
        print(f"Loading dataset: {self.dataset_name}")
        print(f"Subset: {self.subset_name}, Split: {self.split}")

        try:
            # Try loading from ModelScope via ms datasets
            from modelscope.msdatasets import MsDataset

            dataset = MsDataset.load(
                self.dataset_name,
                subset_name=self.subset_name,
                split=self.split
            )
            print(f"Loaded {len(dataset)} samples from ModelScope")
            return dataset

        except Exception as e:
            print(f"ModelScope loading failed: {e}")
            print("Trying HuggingFace datasets...")

            try:
                from datasets import load_dataset
                dataset = load_dataset(
                    self.dataset_name,
                    self.subset_name,
                    split=self.split
                )
                print(f"Loaded {len(dataset)} samples from HuggingFace")
                return dataset
            except Exception as e2:
                print(f"HuggingFace loading also failed: {e2}")
                raise Exception(f"Failed to load dataset from both ModelScope and HuggingFace: {e2}")

    def convert_to_qwen_format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert sample to Qwen3 chat format

        Qwen3 uses a specific chat template format.
        The format typically follows:
        <|im_start|>user
        message<|im_end|>
        <|im_start|>assistant
        response<|im_end|>
        """
        try:
            # Extract conversations from the sample
            # Muice-Dataset format may vary, adapt accordingly
            if 'conversations' in sample:
                conversations = sample['conversations']
            elif 'messages' in sample:
                conversations = sample['messages']
            elif 'content' in sample:
                # Single turn format
                conversations = [
                    {'from': 'user', 'value': sample['content']},
                    {'from': 'assistant', 'value': sample.get('response', sample.get('answer', ''))}
                ]
            else:
                # Try to extract from other fields
                conversations = []
                for key in sample.keys():
                    if key not in ['id', 'source', 'category']:
                        if isinstance(sample[key], str):
                            conversations.append({'from': 'user', 'value': sample[key]})

            # Build Qwen3 format conversation
            formatted_messages = []
            for conv in conversations:
                role = conv.get('from', conv.get('role', 'user'))
                value = conv.get('value', conv.get('content', ''))

                if role == 'user':
                    formatted_messages.append({'role': 'user', 'content': value})
                elif role == 'assistant':
                    formatted_messages.append({'role': 'assistant', 'content': value})
                else:
                    formatted_messages.append({'role': 'user', 'content': str(value)})

            # Use tokenizer's chat template if available
            if self.tokenizer:
                try:
                    # Try standard transformers chat template
                    text = self.tokenizer.apply_chat_template(
                        formatted_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except:
                    # Manual template construction
                    text = self._build_chat_template(formatted_messages)
            else:
                text = self._build_chat_template(formatted_messages)

            return {
                'text': text,
                'original': sample
            }

        except Exception as e:
            print(f"Error converting sample: {e}")
            traceback.print_exc()
            return {'text': '', 'original': sample}

    def _build_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Manually build chat template for Qwen"""
        template = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                template += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                template += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == 'system':
                template += f"<|im_start|>system\n{content}<|im_end|>\n"

        # Add generation prompt for assistant
        template += "<|im_start|>assistant\n"
        return template

    def tokenize_function(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize the formatted text"""
        if not example.get('text'):
            return {'input_ids': [], 'attention_mask': []}

        try:
            result = self.tokenizer(
                example['text'],
                truncation=True,
                max_length=self.max_seq_length,
                padding='max_length',
                return_tensors=None
            )
            return {
                'input_ids': result['input_ids'],
                'attention_mask': result['attention_mask'],
                'labels': result['input_ids'].copy()
            }
        except Exception as e:
            print(f"Tokenization error: {e}")
            return {'input_ids': [], 'attention_mask': [], 'labels': []}

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Full preprocessing pipeline"""
        print("Converting to Qwen format...")
        converted = dataset.map(
            self.convert_to_qwen_format,
            desc="Converting to Qwen format",
            remove_columns=dataset.column_names
        )

        print("Tokenizing...")
        tokenized = converted.map(
            self.tokenize_function,
            desc="Tokenizing",
            batched=False,
            remove_columns=['text']
        )

        # Filter out empty samples
        tokenized = tokenized.filter(
            lambda x: len(x['input_ids']) > 0,
            desc="Filtering empty samples"
        )

        return tokenized

    def save_processed_data(self, dataset: Dataset, output_path: str = None):
        """Save processed dataset"""
        output_path = output_path or self.output_path

        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as JSON for flexibility
        data_list = []
        for item in dataset:
            data_list.append({
                'input_ids': item['input_ids'],
                'attention_mask': item['attention_mask'],
                'labels': item['labels']
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False)

        print(f"Saved {len(data_list)} processed samples to {output_path}")

    def run(self):
        """Main preprocessing pipeline"""
        print("=" * 50)
        print("Starting Data Preprocessing")
        print("=" * 50)

        # Step 1: Load tokenizer
        self.load_tokenizer()

        # Step 2: Load raw data
        raw_data = self.load_raw_data()

        # Step 3: Preprocess
        processed_data = self.preprocess_dataset(raw_data)

        # Step 4: Save
        self.save_processed_data(processed_data)

        print("=" * 50)
        print(f"Preprocessing complete! Processed {len(processed_data)} samples")
        print(f"Output saved to: {self.output_path}")
        print("=" * 50)

        return processed_data


def main():
    """CLI entry point"""
    preprocessor = DataPreprocessor()
    preprocessor.run()


if __name__ == "__main__":
    main()