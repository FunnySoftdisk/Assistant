"""
Deployment Script for Qwen3 using vLLM
Supports API server mode and offline inference
"""

import os
import argparse
from typing import Optional, List, Dict
import subprocess
import signal
import sys

from config.train_config import DEPLOY_CONFIG, setup_environment


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy Qwen3 model using vLLM")

    parser.add_argument("--model_path", type=str,
                        default=DEPLOY_CONFIG["model_path"],
                        help="Path to the model")
    parser.add_argument("--host", type=str,
                        default=DEPLOY_CONFIG["host"],
                        help="Host to bind to")
    parser.add_argument("--port", type=int,
                        default=DEPLOY_CONFIG["port"],
                        help="Port to bind to")
    parser.add_argument("--gpu_memory_utilization", type=float,
                        default=DEPLOY_CONFIG["gpu_memory_utilization"],
                        help="GPU memory utilization ratio (0-1)")
    parser.add_argument("--tensor_parallel_size", type=int,
                        default=DEPLOY_CONFIG["tensor_parallel_size"],
                        help="Tensor parallel size for multi-GPU")
    parser.add_argument("--max_num_seqs", type=int,
                        default=DEPLOY_CONFIG["max_num_seqs"],
                        help="Maximum number of sequences")
    parser.add_argument("--max_model_len", type=int,
                        default=DEPLOY_CONFIG["max_model_len"],
                        help="Maximum model length")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "float32", "bfloat16"],
                        help="Model dtype")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                        help="Trust remote code")
    parser.add_argument("--chat_template", type=str, default=None,
                        help="Chat template file path")

    # Server options
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for authentication")
    parser.add_argument("--response_role", type=str, default="assistant",
                        help="Role for response")
    parser.add_argument("--disable_log_stats", action="store_true", default=False,
                        help="Disable logging stats")

    # Quantization options
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["awq", "gptq", "fp8", None],
                        help="Quantization method")
    parser.add_argument("--enforce_eager", action="store_true", default=False,
                        help="Enforce eager mode (no CUDA graphs)")

    return parser.parse_args()


class DeploymentManager:
    """Manages vLLM deployment lifecycle"""

    def __init__(self, args):
        self.args = args
        self.process = None
        self.server_url = None

    def build_vllm_command(self) -> List[str]:
        """Build vLLM server command"""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.args.model_path,
            "--host", self.args.host,
            "--port", str(self.args.port),
            "--gpu-memory-utilization", str(self.args.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.args.tensor_parallel_size),
            "--max-num-seqs", str(self.args.max_num_seqs),
            "--max-model-len", str(self.args.max_model_len),
            "--dtype", self.args.dtype,
            "--trust-remote-code",
        ]

        if self.args.api_key:
            cmd.extend(["--api-key", self.args.api_key])

        if self.args.chat_template:
            cmd.extend(["--chat-template", self.args.chat_template])

        if self.args.quantization:
            cmd.extend(["--quantization", self.args.quantization])

        if self.args.enforce_eager:
            cmd.append("--enforce-eager")

        if self.args.disable_log_stats:
            cmd.append("--disable-log-stats")

        return cmd

    def start_server(self):
        """Start the vLLM server"""
        print("=" * 60)
        print("Starting vLLM Server")
        print("=" * 60)
        print(f"Model: {self.args.model_path}")
        print(f"Host: {self.args.host}:{self.args.port}")
        print(f"GPU Memory Utilization: {self.args.gpu_memory_utilization}")
        print(f"Tensor Parallel Size: {self.args.tensor_parallel_size}")
        print("=" * 60)

        cmd = self.build_vllm_command()
        print(f"\nCommand: {' '.join(cmd)}\n")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            self.server_url = f"http://{self.args.host}:{self.args.port}"

            # Monitor output
            print("Server output:")
            print("-" * 40)

            for line in self.process.stdout:
                print(line, end="")
                if "Uvicorn running on" in line or "Running on" in line:
                    break

            print("-" * 40)
            print(f"\nServer started at: {self.server_url}")
            print("\nAPI Endpoints:")
            print(f"  - Chat Completions: {self.server_url}/v1/chat/completions")
            print(f"  - Completions: {self.server_url}/v1/completions")
            print(f"  - Models: {self.server_url}/v1/models")

            return True

        except Exception as e:
            print(f"Failed to start server: {e}")
            return False

    def stop_server(self):
        """Stop the vLLM server"""
        if self.process:
            print("\nStopping server...")
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("Server stopped")

    def test_connection(self) -> bool:
        """Test API connection"""
        import urllib.request
        import json

        try:
            url = f"{self.server_url}/v1/models"
            with urllib.request.urlopen(url, timeout=10) as response:
                models = json.loads(response.read().decode())
                print("\nAvailable models:")
                for model in models.get("data", []):
                    print(f"  - {model.get('id', 'unknown')}")
                return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


def test_chat_completion(server_url: str, messages: List[Dict[str, str]]):
    """Test chat completion API"""
    import urllib.request
    import json

    print("\n" + "=" * 60)
    print("Testing Chat Completion")
    print("=" * 60)

    payload = {
        "model": "Qwen",
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": False
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{server_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode())
            print("\nResponse:")
            print(f"Model: {result.get('model', 'unknown')}")
            for choice in result.get("choices", []):
                print(f"Finish reason: {choice.get('finish_reason', 'unknown')}")
                content = choice.get("message", {}).get("content", "")
                print(f"Content: {content[:500]}...")

            return True

    except Exception as e:
        print(f"Chat completion test failed: {e}")
        return False


def main():
    """Entry point"""
    args = parse_args()
    setup_environment()

    manager = DeploymentManager(args)

    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal")
        manager.stop_server()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        print("Please run merge_model.py first or specify correct path")
        sys.exit(1)

    # Start server
    if not manager.start_server():
        sys.exit(1)

    # Wait for server to be ready
    import time
    time.sleep(3)

    # Test connection
    if not manager.test_connection():
        print("Warning: Could not verify server connection")

    # Run interactive demo
    print("\n" + "=" * 60)
    print("Interactive Demo Mode")
    print("=" * 60)
    print("Type your messages and press Enter. Type 'quit' to exit.")
    print("-" * 60)

    messages = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            if test_chat_completion(manager.server_url, messages):
                # Add assistant response to history
                pass

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    # Cleanup
    manager.stop_server()
    print("\nGoodbye!")


if __name__ == "__main__":
    main()