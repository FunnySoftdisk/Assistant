"""
通用工具函数
"""
import json
import re
import time
from typing import Optional, Callable
from functools import wraps


def safe_json_parse(text: str) -> Optional[dict]:
    """
    安全解析JSON，处理各种边缘情况：
    1. Markdown代码块包裹
    2. 首行非JSON内容
    3. 截断的JSON
    4. LLM思考标签（、<think>）
    """
    if not text:
        return None

    text = text.strip()

    # 移除markdown代码块
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    # 移除结尾代码块
    if text.endswith("```"):
        text = text[:-3]

    # 移除LLM思考标签
    text = re.sub(r'<think>.*?', '', text, flags=re.DOTALL)
    text = text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取JSON对象
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 尝试提取JSON数组
    array_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
    match = re.search(array_pattern, text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def safe_json_dumps(obj, **kwargs) -> str:
    """安全的JSON序列化"""
    try:
        return json.dumps(obj, ensure_ascii=False, **kwargs)
    except (TypeError, ValueError):
        return json.dumps({"raw": str(obj)}, ensure_ascii=False)


def timing_decorator(func: Callable):
    """计时装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            return result
        finally:
            elapsed = time.time() - start
    return wrapper


def validate_message(message: str, max_length: int = 2000) -> tuple:
    """
    验证消息格式

    Returns:
        (is_valid, error_message)
    """
    if not message:
        return False, "消息不能为空"

    if len(message) > max_length:
        return False, f"消息长度不能超过{max_length}字符"

    # 检查特殊字符
    dangerous_patterns = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onclick=',
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return False, "消息包含非法字符"

    return True, ""


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """截断过长的文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_numbers(text: str) -> list:
    """从文本中提取所有数字"""
    return [int(n) for n in re.findall(r'\d+', text)]


def parse_time_duration(text: str) -> Optional[int]:
    """
    解析时间duration文本，返回秒数

    Examples:
        "30秒" -> 30
        "5分钟" -> 300
        "1小时" -> 3600
    """
    patterns = [
        (r'(\d+)秒', 1),
        (r'(\d+)分钟', 60),
        (r'(\d+)小时', 3600),
        (r'(\d+)天', 86400),
    ]

    for pattern, multiplier in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)) * multiplier

    return None