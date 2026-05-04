"""
通用工具函数
"""
import json
import re
from typing import Optional


def safe_json_parse(text: str) -> Optional[dict]:
    """
    安全解析JSON，处理各种边缘情况：
    1. Markdown代码块包裹
    2. 首行非JSON内容
    3. 截断的JSON
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
