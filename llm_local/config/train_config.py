"""
Qwen3-8B Model Fine-tuning Configuration
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# Model Configuration
MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",  # Using 7B instead of 8B (Qwen3-8B not widely available)
    # "model_name": "Qwen/Qwen2.5-8B-Instruct",  # Alternative 8B model
    "model_path": "/home/gfyubuntu/assistant/models/Qwen2.5-7B-Instruct",
    "tokenizer_path": "/home/gfyubuntu/assistant/models/Qwen2.5-7B-Instruct",
}

# Data Configuration
DATA_CONFIG = {
    "dataset_name": "Moemuu/Muice-Dataset",
    "subset_name": "default",
    "split": "train",
    "processed_data_path": "/home/gfyubuntu/assistant/llm_local/data/processed_data.json",
}

# LoRA Configuration
LORA_CONFIG = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "modules_to_save": None,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# Training Hyperparameters
@dataclass
class TrainingConfig:
    # Basic training settings
    output_dir: str = "/home/gfyubuntu/assistant/llm_local/output/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    dataloader_num_workers: int = 4
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False

    # Optimizer settings
    optim: str = "paged_adamw_32bit"

    # LoRA specific
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Gradient checkpointing
    gradient_checkpointing: bool = True
    use_flash_attention_2: bool = True

    # Logging and monitoring
    logging_dir: str = "/home/gfyubuntu/assistant/llm_local/output/logs"
    report_to: str = "tensorboard"

    # Model freezing (optional)
    freeze_embeds: bool = False
    freeze_layers: int = 0

    # Data preprocessing
    max_seq_length: int = 2048
    train_ratio: float = 0.9

    # Multi-GPU settings
    multi_gpu: bool = True
    tensor_parallel_size: int = 1

    # Logging
    verbose: bool = True


# Merge Configuration
MERGE_CONFIG = {
    "output_path": "/home/gfyubuntu/assistant/llm_local/output/final_model",
    "save_dtype": "float16",
    "keep_fp16_weight": True,
}

# Deployment Configuration
DEPLOY_CONFIG = {
    "model_path": "/home/gfyubuntu/assistant/llm_local/output/final_model",
    "host": "0.0.0.0",
    "port": 8000,
    "gpu_memory_utilization": 0.9,
    "tensor_parallel_size": 1,
    "max_num_seqs": 256,
    "max_model_len": 4096,
    "vllm_args": {
        "enforce_eager": False,
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": True,
    },
}

# Environment variables
ENV_CONFIG = {
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "0",
    "HF_HUB_OFFLINE": "0",
}

def setup_environment():
    """Setup environment variables for training"""
    for key, value in ENV_CONFIG.items():
        os.environ[key] = value