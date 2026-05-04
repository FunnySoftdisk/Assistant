"""
Merge LoRA weights into base model
Supports loading from checkpoint and saving the merged model
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel, PeftConfig

from config.train_config import (
    MODEL_CONFIG,
    MERGE_CONFIG,
    setup_environment
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Merge LoRA weights to base model")

    parser.add_argument("--base_model_path", type=str,
                        default=MODEL_CONFIG["model_path"],
                        help="Path to base model")
    parser.add_argument("--peft_model_path", type=str,
                        default="/home/gfyubuntu/assistant/llm_local/output/checkpoints/final",
                        help="Path to LoRA/PEFT model")
    parser.add_argument("--output_path", type=str,
                        default=MERGE_CONFIG["output_path"],
                        help="Output path for merged model")
    parser.add_argument("--save_dtype", type=str,
                        default=MERGE_CONFIG["save_dtype"],
                        choices=["float16", "float32", "bfloat16"],
                        help="Data type for saving")
    parser.add_argument("--safe_serialization", action="store_true", default=True,
                        help="Use safe serialization (safetensors)")
    parser.add_argument("--tokenizer_only", action="store_true", default=False,
                        help="Only save tokenizer")

    return parser.parse_args()


def get_dtype(dtype_str: str):
    """Convert string to torch dtype"""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float16)


def merge_models(args):
    """Merge LoRA weights into base model"""
    setup_environment()

    print("=" * 60)
    print("Merging LoRA weights with Base Model")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Load base model
    print(f"\nLoading base model from: {args.base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=get_dtype(args.save_dtype),
        trust_remote_code=True,
        device_map="cpu",  # Load on CPU first for merging
    )
    base_model.config.use_cache = True
    print(f"Base model loaded, dtype: {base_model.dtype}")

    # Load LoRA model
    print(f"\nLoading LoRA model from: {args.peft_model_path}")

    # Check if it's a PEFT model or a full checkpoint
    peft_config_path = os.path.join(args.peft_model_path, "adapter_config.json")
    if os.path.exists(peft_config_path):
        print("Detected PEFT adapter format")
        # Load as PEFT model
        peft_config = PeftConfig.from_pretrained(args.peft_model_path)
        model = PeftModel.from_pretrained(
            base_model,
            args.peft_model_path,
            is_trainable=False
        )
    else:
        print("Loading as full model...")
        model = base_model

    # Merge weights
    print("\nMerging LoRA weights...")
    if hasattr(model, 'merge_and_unload'):
        print("Using merge_and_unload() method")
        merged_model = model.merge_and_unload()
    else:
        print("Using direct parameter merging")
        merged_model = model

    # Ensure dtype
    merged_model = merged_model.to(dtype=get_dtype(args.save_dtype))

    print(f"Merged model dtype: {merged_model.dtype}")

    # Save merged model
    print(f"\nSaving merged model to: {args.output_path}")
    merged_model.save_pretrained(
        args.output_path,
        safe_serialization=args.safe_serialization,
    )

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.save_pretrained(args.output_path)

    # Save config
    print("Saving config files...")
    merged_model.config.save_pretrained(args.output_path)

    print("\n" + "=" * 60)
    print("Model merge completed successfully!")
    print(f"Output path: {args.output_path}")
    print("=" * 60)


def verify_merged_model(model_path: str):
    """Verify the merged model can be loaded correctly"""
    print("\nVerifying merged model...")

    try:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
        )
        print(f"Model loaded successfully")
        print(f"Model type: {type(model)}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        print(f"Tokenizer loaded successfully")

        # Test generation
        print("\nTesting generation...")
        device = model.device
        input_text = "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Generation test passed!")
        print(f"Response: {response[:200]}...")

        return True

    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def main():
    """Entry point"""
    args = parse_args()

    try:
        merge_models(args)

        # Verify if not tokenizer_only
        if not args.tokenizer_only:
            verify_merged_model(args.output_path)

    except Exception as e:
        print(f"\nMerge failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()