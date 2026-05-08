"""
意图识别Agent - 使用真实LLM进行语义理解
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
import re
import asyncio
from typing import Optional, Union, List, Dict

from core.llm_client import llm_chat, LLMResponse, LLMTimeoutError, LLMError
from core.utils import safe_json_parse, validate_message


class IntentionAgent(AgentBase):
    """
    意图识别Agent - 基于LLM语义理解识别用户意图
    支持7大类意图识别 + 实体提取

    特性:
    - LLM超时保护（30秒）
    - 关键词回退机制（LLM失败时）
    - 输入验证和长度控制
    - 异常恢复
    """

    # 意图类型定义
    INTENT_TYPES = [
        "travel_planning",    # 行程规划
        "memory_query",       # 记忆查询
        "preference_manage",  # 偏好管理
        "info_query",         # 信息查询
        "event_collection",  # 事项收集
        "execution",          # 执行操作
        "general_chat"       # 一般对话
    ]

    # 系统提示词
    SYSTEM_PROMPT = """你是一个智能助手，负责识别用户的意图。

支持的意图类型：
- travel_planning: 规划旅行行程，如"帮我规划去上海的行程"
- memory_query: 查询历史记忆，如"我之前去过哪里"
- preference_manage: 管理用户偏好，如"我喜欢汉庭酒店"
- info_query: 信息查询，如"今天天气怎么样"、"搜索xxx"
- event_collection: 收集行程要素，如"我3月5日去北京"
- execution: 执行操作，如"帮我做xxx"
- general_chat: 一般对话，如"你好"、"谢谢"

请分析用户输入，返回JSON格式的意图识别结果：
{
    "intent": "意图类型",
    "confidence": 0.95,
    "entities": {"locations": ["上海"], "date": "3月5日"},
    "reasoning": "简单推理说明"
}

confidence: 0-1之间的置信度
entities: 提取的关键信息（地点、时间、人名等）

注意：如果输入包含【对话历史】，请结合上下文理解用户意图。"""

    def __init__(self, name: str = "IntentionAgent", model_config: dict = None, **kwargs):
        super().__init__()
        self.name = name
        self.model_config = model_config or {}
        self.memory = InMemoryMemory()

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """识别用户意图 - 使用LLM"""
        if x is None:
            return Msg(name=self.name, content=json.dumps({"error": "No input"}), role="assistant")

        if isinstance(x, list):
            x = x[-1]

        query = x.content if hasattr(x, 'content') else str(x)

        # 验证输入
        is_valid, error_msg = validate_message(query)
        if not is_valid:
            return Msg(
                name=self.name,
                content=json.dumps({
                    "intent": "general_chat",
                    "confidence": 0.0,
                    "entities": {},
                    "error": error_msg,
                    "query": query
                }, ensure_ascii=False),
                role="assistant"
            )

        # 调用LLM进行意图识别
        try:
            result = await self._classify_intent(query)
        except asyncio.TimeoutError:
            # LLM超时，使用关键词回退
            result = self._classify_by_keywords(query)
            result["fallback_reason"] = "LLM timeout"
        except Exception as e:
            # 发生异常，使用关键词回退
            result = self._classify_by_keywords(query)
            result["fallback_reason"] = f"Error: {str(e)}"

        return Msg(
            name=self.name,
            content=json.dumps(result, ensure_ascii=False),
            role="assistant"
        )

    async def _classify_intent(self, query: str) -> Dict:
        """使用LLM进行意图分类和实体提取"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]

        try:
            # 添加超时保护
            response = await asyncio.wait_for(
                llm_chat(messages),
                timeout=30.0
            )

            # 解析JSON响应（安全解析）
            result = safe_json_parse(response)

            if result is None:
                # 解析失败，使用关键词回退
                return self._classify_by_keywords(query)

            # 验证intent有效性
            if result.get("intent") not in self.INTENT_TYPES:
                result["intent"] = "general_chat"
                result["confidence"] = 0.5

            # 确保entities存在
            if "entities" not in result:
                result["entities"] = {}

            # 补充实体提取
            result["entities"].update(self._extract_entities_fallback(query))
            result["query"] = query

            return result

        except asyncio.TimeoutError:
            # 重试一次
            try:
                response = await asyncio.wait_for(
                    llm_chat([{"role": "user", "content": f"识别意图: {query}"}]),
                    timeout=30.0
                )
                result = safe_json_parse(response)
                if result and result.get("intent") in self.INTENT_TYPES:
                    return result
            except:
                pass

            raise  # 让外层捕获并使用关键词回退

        except json.JSONDecodeError:
            # LLM返回非JSON，使用关键词回退
            return self._classify_by_keywords(query)
        except LLMError as e:
            # LLM调用错误，记录日志并使用关键词回退
            return self._classify_by_keywords(query)

    def _classify_by_keywords(self, query: str) -> Dict:
        """关键词回退意图分类"""
        query_lower = query.lower()

        intent_mapping = {
            "travel_planning": ["规划", "行程", "去", "旅游", "出差", "旅行", " trip", "travel"],
            "memory_query": ["查询", "记忆", "之前", "历史", "去过", "做过", "上次"],
            "preference_manage": ["喜欢", "偏好", "习惯", "讨厌", "爱", " preferences"],
            "info_query": ["搜索", "查询", "什么是", "怎么样", "天气", "search", "天气"],
            "event_collection": ["收集", "确认", "出发", "时间", "日期"],
            "execution": ["执行", "操作", "帮", "做", "帮我"]
        }

        scores = {}
        for intent, keywords in intent_mapping.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[intent] = score

        if not scores:
            return {
                "intent": "general_chat",
                "confidence": 0.5,
                "entities": self._extract_entities_fallback(query),
                "query": query,
                "fallback": True
            }

        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent] / 3.0, 1.0)

        return {
            "intent": best_intent,
            "confidence": confidence,
            "entities": self._extract_entities_fallback(query),
            "query": query,
            "fallback": True
        }

    def _extract_entities_fallback(self, query: str) -> Dict:
        """使用正则表达式提取实体"""
        entities = {}

        # 时间提取
        time_patterns = [
            (r'\d+月\d+日', 'date'),
            (r'\d+年\d+月\d+日', 'date'),
            (r'今天|明天|后天', 'relative_date'),
            (r'下个?(周|个月)', 'future'),
            (r'这周|下周', 'week'),
        ]
        for pattern, key in time_patterns:
            match = re.search(pattern, query)
            if match:
                entities[key] = match.group()

        # 地点提取
        locations = ["北京", "上海", "杭州", "深圳", "广州", "成都", "重庆", "武汉", "西安", "南京"]
        found_locations = [loc for loc in locations if loc in query]
        if found_locations:
            entities["locations"] = found_locations

        # 数字提取
        number_match = re.search(r'\d+(?:人|天|小时|分钟)?', query)
        if number_match:
            entities["number"] = number_match.group()

        return entities