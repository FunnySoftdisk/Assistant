# Qwen3-8B Fine-tuning Project

A complete project for fine-tuning Qwen3-8B model using LoRA/QLoRA technique with support for multi-GPU training and vLLM deployment.

## Project Structure

```
llm_local/
├── config/
│   └── train_config.py      # Configuration file for all settings
├── data/
│   └── processed_data.json  # Processed training data (after preprocessing)
├── output/
│   ├── checkpoints/         # Training checkpoints
│   ├── final_model/         # Merged final model
│   └── logs/                # Training logs
├── data_preprocessing.py    # Data loading and preprocessing
├── train.py                 # Training script with LoRA
├── merge_model.py           # Script to merge LoRA weights
├── deploy.py                # vLLM deployment script
└── README.md                # This file
```

## Requirements

```bash
# Core dependencies
pip install torch transformers datasets peft
pip install modelscope ms-swift
pip install vllm

# Optional but recommended
pip install accelerate bitsandbytes
pip install tensorboard deepspeed

# For data processing
pip install tqdm
```

## Quick Start

### 1. Data Preprocessing

```bash
cd /home/gfyubuntu/assistant/llm_local
python data_preprocessing.py
```

This will:
- Load dataset from ModelScope
- Convert to Qwen3 chat format
- Tokenize and save to `data/processed_data.json`

### 2. Training

Single GPU:
```bash
python train.py \
    --model_path /home/gfyubuntu/assistant/models/Qwen2.5-7B-Instruct \
    --output_dir ./output/checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lora_rank 16
```

Multi-GPU (using torchrun):
```bash
torchrun --nproc_per_node=2 train.py \
    --model_path /home/gfyubuntu/assistant/models/Qwen2.5-7B-Instruct \
    --output_dir ./output/checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4
```

Resume from checkpoint:
```bash
python train.py \
    --resume_from_checkpoint ./output/checkpoints/checkpoint-1000
```

### 3. Merge Model

After training, merge LoRA weights into the base model:

```bash
python merge_model.py \
    --base_model_path /home/gfyubuntu/assistant/models/Qwen2.5-7B-Instruct \
    --peft_model_path ./output/checkpoints/final \
    --output_path ./output/final_model
```

### 4. Deployment

Deploy using vLLM:

```bash
python deploy.py \
    --model_path ./output/final_model \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu_memory_utilization 0.9
```

Or use the vLLM CLI directly:

```bash
vllm serve ./output/final_model \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9
```

## Configuration

All configurations are centralized in `config/train_config.py`:

### Model Configuration
```python
MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "model_path": "/path/to/model",
    "tokenizer_path": "/path/to/tokenizer",
}
```

### LoRA Configuration
```python
LORA_CONFIG = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", ...],
}
```

### Training Configuration
```python
@dataclass
class TrainingConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    max_seq_length: int = 2048
    # ... more options
```

## API Usage

After deployment, use the OpenAI-compatible API:

### Chat Completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # Not required for local deployment
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="Qwen",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=256,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Advanced Features

### QLoRA (Quantized LoRA)

For reduced memory usage with 4-bit quantization:

```python
# In train.py, modify model loading:
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Multi-GPU Training

```bash
# Using torchrun for proper multi-GPU setup
torchrun --nproc_per_node=2 train.py \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4
```

### Custom Dataset

To use your own dataset, modify `data_preprocessing.py`:

```python
def convert_to_qwen_format(self, sample):
    # Convert your dataset format to Qwen format
    return {
        'text': "<|im_start|>user\n...",
        'original': sample
    }
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `--per_device_train_batch_size 1`
2. Increase gradient accumulation: `--gradient_accumulation_steps 16`
3. Enable QLoRA with 4-bit quantization
4. Use `fp16` instead of `bf16`

### Model Not Found

Ensure the model is downloaded:
```python
from modelscope import snapshot_download
model_dir = snapshot_download("Qwen/Qwen2.5-7B-Instruct")
```

### vLLM Server Issues

1. Check GPU availability: `nvidia-smi`
2. Reduce `gpu_memory_utilization`
3. Enable `enforce_eager` mode
4. Set `trust_remote_code=True`

## Monitoring

Training metrics can be viewed with TensorBoard:

```bash
tensorboard --logdir ./output/logs
```

Or use the wandb integration by setting `report_to: "wandb"` in config.

## License

This project follows the license of the base model Qwen2.5.