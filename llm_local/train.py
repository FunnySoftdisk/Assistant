"""
Training Script for Qwen3-8B using LoRA/QLoRA fine-tuning
Supports single and multi-GPU training
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)
from datasets import load_dataset, Dataset
import bitsandbytes as bnb

# Import configuration
from config.train_config import (
    MODEL_CONFIG,
    DATA_CONFIG,
    LORA_CONFIG,
    TrainingConfig,
    setup_environment
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Qwen3 with LoRA")

    # Model arguments
    parser.add_argument("--model_path", type=str, default=MODEL_CONFIG["model_path"],
                        help="Path to the base model")
    parser.add_argument("--tokenizer_path", type=str, default=MODEL_CONFIG["tokenizer_path"],
                        help="Path to the tokenizer")

    # Data arguments
    parser.add_argument("--data_path", type=str, default=DATA_CONFIG["processed_data_path"],
                        help="Path to processed training data")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=LORA_CONFIG["lora_rank"],
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=LORA_CONFIG["lora_alpha"],
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=LORA_CONFIG["lora_dropout"],
                        help="LoRA dropout")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="/home/gfyubuntu/assistant/llm_local/output/checkpoints",
                        help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Save total limit")

    # Mixed precision
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use BF16")

    # Multi-GPU
    parser.add_argument("--multi_gpu", action="store_true", default=True,
                        help="Use multiple GPUs")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint path")

    return parser.parse_args()


def setup_model_and_tokenizer(args):
    """Setup model and tokenizer with appropriate precision"""
    print("=" * 60)
    print("Setting up Model and Tokenizer")
    print("=" * 60)

    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.chat_template is None:
        # Set default chat template for Qwen
        tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # Load model
    print(f"Loading model from: {args.model_path}")

    load_kwargs = {
        "trust_remote_code": True,
    }

    # Configure dtype based on hardware
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported() and args.bf16:
            load_kwargs["torch_dtype"] = torch.bfloat16
            print("Using BF16")
        elif args.fp16:
            load_kwargs["torch_dtype"] = torch.float16
            print("Using FP16")
        else:
            load_kwargs["torch_dtype"] = torch.float16
            print("Using FP16 (default)")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **load_kwargs
    )

    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))

    print(f"Model loaded successfully!")
    print(f"Model dtype: {model.dtype}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    return model, tokenizer


def setup_lora_config(args):
    """Setup LoRA configuration"""
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_CONFIG["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=None
    )
    return lora_config


def prepare_dataset(args, tokenizer):
    """Load and prepare dataset"""
    print("=" * 60)
    print("Loading and Preparing Dataset")
    print("=" * 60)

    # Load processed data
    if os.path.exists(args.data_path):
        print(f"Loading data from {args.data_path}")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(data_list)
    else:
        # Try loading raw dataset
        print("Processed data not found, loading raw dataset...")
        from modelscope.msdatasets import MsDataset
        raw_dataset = MsDataset.load(
            DATA_CONFIG["dataset_name"],
            subset_name=DATA_CONFIG["subset_name"],
            split=DATA_CONFIG["split"]
        )

        # Process on-the-fly
        from data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        preprocessor.tokenizer = tokenizer
        dataset = preprocessor.preprocess_dataset(raw_dataset)

    # Split train/eval
    if args.num_train_epochs > 0:
        split_dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def setup_training_arguments(args):
    """Setup training arguments"""
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_dir="/home/gfyubuntu/assistant/llm_local/output/logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="tensorboard",
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=False,
        group_by_length=False,
        length_column_name="length",
        prediction_loss_only=False,
        hub_token=None,
        push_to_hub=False,
        save_safetensors=True,
        seed=args.seed,
        local_rank=-1,
    )
    return training_args


def train_model(args):
    """Main training function"""
    # Setup environment
    setup_environment()
    set_seed(args.seed)

    # Set CUDA device
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args)

    # Apply LoRA
    print("=" * 60)
    print("Applying LoRA")
    print("=" * 60)
    lora_config = setup_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(args, tokenizer)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    # Setup training arguments
    training_args = setup_training_arguments(args)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Print training info
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of epochs: {args.num_train_epochs}")
    print(f"Per device batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"FP16: {args.fp16}")
    print(f"BF16: {args.bf16}")
    print("=" * 60)

    # Resume training if specified
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")

    # Start training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    print("Saving final model...")
    final_output_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_output_dir)
    trainer.save_state()

    print("=" * 60)
    print("Training completed!")
    print(f"Model saved to: {final_output_dir}")
    print("=" * 60)


def main():
    """Entry point"""
    args = parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()