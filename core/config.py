"""
项目配置文件
"""
import os

# ============================================================
# AgentScope配置
# ============================================================
AGENTSCOPE_CONFIG = {
    "project": "Multi-Agent-Assistant",
    "name": "main_assistant",
    "logging_level": "INFO"
}

# ============================================================
# LLM配置 - 支持Qwen3.5 (阿里云百炼) 或 OpenAI兼容API
# ============================================================

# 方式1: 阿里云百炼/Qwen3.5 API
#   - 获取地址: https://bailian.console.aliyun.com/
#   - 模型名称: qwen-turbo, qwen-plus, qwen-max, qwen-long 等
DASHSCOPE_CONFIG = {
    # "api_key": 
    
    # 模型名称: qwen-turbo, qwen-plus, qwen-max, qwen-long 等
    "model_name": os.getenv("DASHSCOPE_MODEL", "qwen3-8b"),
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "temperature": 0.7,
    "max_tokens": 2000,
    # Qwen3系列需要设置enable_thinking为false（非思考模式）
    "extra_params": {
        "enable_thinking": False
    }
}

# 方式2: OpenAI兼容API (如vllm本地部署)
#   - vllm部署后通常是 http://localhost:8000/v1
OPENAI_COMPAT_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", "not-required"),
    "model_name": os.getenv("MODEL_NAME", "your-model-name"),
    "base_url": os.getenv("BASE_URL", "http://localhost:8000/v1"),
    "temperature": 0.7,
    "max_tokens": 2000,
}

# ============================================================
# 选择使用的LLM后端
#   - "dashscope": 阿里云百炼 (Qwen3.5)
#   - "openai": OpenAI兼容API (vllm本地部署)
#   - "local": 本地vllm服务
# ============================================================
LLM_BACKEND = os.getenv("LLM_BACKEND", "dashscope")  # 默认使用Qwen3.5

def get_llm_config():
    """获取当前配置的LLM配置"""
    if LLM_BACKEND == "dashscope":
        return DASHSCOPE_CONFIG
    elif LLM_BACKEND == "openai":
        return OPENAI_COMPAT_CONFIG
    else:
        return DASHSCOPE_CONFIG  # 默认

# ============================================================
# 调度配置
# ============================================================
SCHEDULER_CONFIG = {
    "priority_levels": {
        1: ["preference_agent", "info_query_agent", "execution_agent", "tool_skill"],
        2: ["planning_agent", "summarization_agent"],
    },
    "timeout": 30,
    "max_parallel_tasks": 5,
}

# ============================================================
# 记忆配置
# ============================================================
MEMORY_CONFIG = {
    "short_term_ttl": 3600,  # 1小时
    "max_history": 20,
    "storage_dir": "data/memory",
}

# ============================================================
# 外部执行配置
# ============================================================
EXECUTION_CONFIG = {
    "max_retries": 3,
    "retry_delay": 1,
    "timeout": 15,
}

# ============================================================
# vLLM本地部署配置 (备用)
# ============================================================
VLLM_CONFIG = {
    "host": "localhost",
    "port": 8000,
    "model_name": "your-trained-model",  # 你本地训练的模型名称
    "gpu_memory_utilization": 0.9,
    "max_model_len": 4096,
}