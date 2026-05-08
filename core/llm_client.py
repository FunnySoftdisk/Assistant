"""
LLM客户端模块 - 统一管理模型调用
支持: 阿里云百炼(Qwen)、OpenAI兼容、vLLM本地部署
"""
import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import httpx


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str


class LLMClient:
    """
    统一LLM客户端
    支持多种后端: dashscope(Qwen), minimax, openai, vllm
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = None,
        base_url: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60,
        extra_params: dict = None,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "not-required")
        self.model_name = model_name or os.getenv("MODEL_NAME", "qwen3-8b")
        self.base_url = base_url or os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_params = extra_params or {}

        # 创建httpx异步客户端
        self.client = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> LLMResponse:
        """
        发送对话请求

        Args:
            messages: [{"role": "user", "content": "..."}]
            model: 可选，覆盖默认模型
            temperature: 可选，覆盖默认温度
            max_tokens: 可选，覆盖默认最大token

        Returns:
            LLMResponse对象
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or self.model_name,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            **self.extra_params,
            **kwargs
        }

        # 过滤None值
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                model=data.get("model", self.model_name),
                usage=data.get("usage", {}),
                finish_reason=data["choices"][0].get("finish_reason", "stop")
            )

        except httpx.HTTPStatusError as e:
            raise LLMError(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise LLMError(f"Request failed: {str(e)}")

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        **kwargs
    ):
        """
        流式对话请求
        yields: str (增量内容)
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **kwargs
        }

        async with self.client.stream("POST", url, headers=headers, json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    if line.strip() == "data: [DONE]":
                        break
                    try:
                        data = json.loads(line[6:])
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except:
                        pass

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()


class LLMError(Exception):
    """LLM调用错误"""
    pass


class LLMTimeoutError(LLMError):
    """LLM调用超时错误"""
    pass


# ============================================================
# 全局LLM客户端实例
# ============================================================
_global_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取全局LLM客户端"""
    global _global_client
    if _global_client is None:
        from core.config import get_llm_config, LLM_BACKEND

        config = get_llm_config()
        _global_client = LLMClient(
            api_key=config["api_key"],
            model_name=config["model_name"],
            base_url=config["base_url"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            extra_params=config.get("extra_params", {}),
        )
        print(f"✓ LLM客户端初始化完成 (backend: {LLM_BACKEND}, model: {config['model_name']})")

    return _global_client


async def close_llm_client():
    """关闭全局LLM客户端"""
    global _global_client
    if _global_client:
        await _global_client.close()
        _global_client = None


# ============================================================
# 便捷函数
# ============================================================

async def llm_chat(messages: List[Dict[str, str]], **kwargs) -> str:
    """简单对话请求"""
    client = get_llm_client()
    response = await client.chat(messages, **kwargs)
    return response.content


async def llm_json(messages: List[Dict[str, str]], **kwargs) -> dict:
    """返回JSON格式的对话请求"""
    client = get_llm_client()

    # 添加格式要求
    system_msg = {
        "role": "system",
        "content": "请以JSON格式回复，不要包含其他文字。JSON格式: {\"key\": \"value\"}"
    }
    messages_with_format = [system_msg] + messages if messages[0]["role"] != "system" else messages

    response = await client.chat(messages_with_format, **kwargs)

    try:
        return json.loads(response.content)
    except:
        return {"raw": response.content}


async def llm_stream(messages: List[Dict[str, str]], **kwargs):
    """流式对话请求"""
    client = get_llm_client()
    async for chunk in client.stream_chat(messages, **kwargs):
        yield chunk