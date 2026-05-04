"""
记忆更新Agent - 智能记忆管理
使用LLM分析对话内容，提取偏好，决定是否存储到长期记忆
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import time

from core.llm_client import llm_chat
from memory.short_term import ShortTermMemory, ConversationState
from memory.long_term import LongTermMemory, UserPreference, TravelHistory


@dataclass
class MemoryResult:
    """记忆操作结果"""
    action: str  # "store_preference", "update_travel", "update_summary", "skip"
    success: bool
    stored_data: Optional[Dict] = None
    response: str = ""
    confidence: float = 0.0


class MemoryAgent(AgentBase):
    """
    记忆更新Agent - 智能记忆管理
    职责：
    1. 分析对话内容，判断是否需要存储
    2. 提取偏好信息并结构化
    3. 决定存储策略（追加/覆盖）
    4. 管理短期记忆（Redis）和长期记忆（PostgreSQL）
    """

    # 系统提示词 - 用于LLM分析
    SYSTEM_PROMPT = """你是一个记忆管理助手，负责分析对话内容，提取用户偏好信息。

你的职责：
1. 判断用户是否在表达偏好（如"我喜欢汉庭"、"我一般坐地铁"）
2. 判断用户是否对某个建议表示满意（如"好的"、"可以"、"就用这个"）
3. 提取偏好信息的结构和值

偏好类别：
- hotel: 酒店品牌（汉庭、如家、万豪等）
- airline: 航空公司（国航、东航等）
- seat: 座位偏好（靠窗、靠过道）
- food: 餐饮偏好（中餐、火锅、海鲜等）
- transport: 交通方式（地铁、打车、公交）
- budget: 预算等级（经济型、舒适型、高端型）

输出JSON格式：
{
    "is_preference": true/false,  // 用户是否在表达偏好
    "is_satisfaction": true/false, // 用户是否对建议表示满意
    "preferences": [
        {"category": "hotel", "key": "brand", "value": "汉庭", "confidence": 0.9}
    ],
    "reasoning": "简单推理说明"
}

confidence: 0-1之间的置信度，表示这个偏好信息的可靠程度"""

    def __init__(self, name: str = "MemoryAgent", model_config: dict = None, **kwargs):
        super().__init__()
        self.name = name
        self.model_config = model_config or {}

        # 初始化记忆系统
        self.short_term = ShortTermMemory()  # Redis存储
        self.long_term = LongTermMemory()    # PostgreSQL存储

        # 尝试初始化数据库
        self.long_term.init_database()

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """处理记忆更新请求"""
        if x is None:
            return Msg(name=self.name, content=json.dumps({"error": "No input"}), role="assistant")

        if isinstance(x, list):
            x = x[-1]

        try:
            data = json.loads(x.content) if isinstance(x.content, str) else x.content
        except:
            data = {"content": str(x.content)}

        # 分析并更新记忆
        result = await self._analyze_and_update_memory(data)

        # 转换为可序列化字典
        result_dict = {
            "action": result.action,
            "success": result.success,
            "stored_data": result.stored_data,
            "response": result.response,
            "confidence": result.confidence
        }

        return Msg(
            name=self.name,
            content=json.dumps(result_dict, ensure_ascii=False),
            role="assistant"
        )

    async def _analyze_and_update_memory(self, data: dict) -> MemoryResult:
        """
        核心记忆分析逻辑
        1. 提取用户输入和Agent响应
        2. 使用LLM分析是否需要存储
        3. 根据分析结果更新短/长期记忆
        """
        user_input = data.get("content", "")
        session_id = data.get("session_id", "default")
        user_id = data.get("user_id", "default")
        intent = data.get("intent", "")
        entities = data.get("entities", {})

        # Step 1: 分析对话内容
        analysis = await self._analyze_conversation(user_input, "", intent)

        stored_data = {}

        # Step 2: 存储偏好（如果有）
        if analysis.get("is_preference") and analysis.get("preferences"):
            success = await self._store_preferences(
                user_id, analysis["preferences"]
            )
            if success:
                stored_data["preferences"] = analysis["preferences"]

        # Step 3: 存储行程历史（如果是旅行规划）
        if intent == "travel_planning" and entities.get("locations"):
            history = TravelHistory(
                user_id=user_id,
                destination=entities["locations"][0],
                start_date=entities.get("date", ""),
                end_date=entities.get("end_date", ""),
                purpose=entities.get("purpose", ""),
                preferences=analysis.get("preferences", [])
            )
            self.long_term.save_travel_history(history)
            stored_data["travel_history"] = {
                "destination": history.destination,
                "date": history.start_date
            }

        # Step 4: 更新短期记忆
        await self._update_short_term(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            intent=intent,
            entities=entities,
            analysis=analysis
        )

        # 生成响应
        response = self._generate_memory_response(analysis, stored_data)

        return MemoryResult(
            action="memory_update",
            success=True,
            stored_data=stored_data,
            response=response,
            confidence=analysis.get("confidence", 0.5)
        )

    async def _analyze_conversation(
        self,
        user_input: str,
        agent_response: str,
        intent: str
    ) -> Dict:
        """使用LLM分析对话内容"""
        # 检查是否是简单的行程保存（不需要LLM分析）
        if intent == "travel_planning":
            return {
                "is_preference": False,
                "is_satisfaction": False,
                "preferences": [],
                "confidence": 1.0,
                "reasoning": "行程规划意图，直接存储"
            }

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"用户输入: {user_input}\n助手回复: {agent_response}\n当前意图: {intent}"}
        ]

        try:
            response = await llm_chat(messages)
            result = json.loads(response)
            return result
        except Exception as e:
            # 分析失败，使用关键词回退
            return self._keyword_based_analysis(user_input)

    def _keyword_based_analysis(self, text: str) -> Dict:
        """基于关键词的简单分析（回退方案）"""
        text_lower = text.lower()

        # 偏好关键词
        preference_keywords = {
            "hotel": ["酒店", "汉庭", "如家", "7天", "万豪", "希尔顿"],
            "airline": ["国航", "东航", "南航", "海航", "航空"],
            "food": ["火锅", "川菜", "粤菜", "中餐", "西餐", "海鲜"],
            "transport": ["地铁", "打车", "公交", "开车", "租车"],
        }

        preferences = []
        for category, keywords in preference_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    preferences.append({
                        "category": category,
                        "key": "keyword_match",
                        "value": kw,
                        "confidence": 0.6
                    })

        return {
            "is_preference": len(preferences) > 0,
            "is_satisfaction": any(word in text_lower for word in ["好", "可以", "行", "用这个", "满意"]),
            "preferences": preferences,
            "confidence": 0.5 if preferences else 0.0,
            "reasoning": "关键词匹配"
        }

    async def _store_preferences(self, user_id: str, preferences: List[Dict]) -> bool:
        """存储偏好到长期记忆"""
        success = True
        for pref in preferences:
            preference = UserPreference(
                user_id=user_id,
                category=pref.get("category", "general"),
                key=pref.get("key", "value"),
                value=pref.get("value", ""),
                confidence=pref.get("confidence", 0.8),
                source="conversation"
            )
            result = self.long_term.save_preference(preference)
            if not result:
                success = False

        # 更新后使缓存失效
        if success:
            self.short_term.invalidate_preferences_cache(user_id)

        return success

    async def _update_short_term(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        intent: str,
        entities: dict,
        analysis: Dict
    ) -> None:
        """更新短期记忆"""
        # 获取当前状态
        state = self.short_term.get_conversation_state(session_id)
        if state is None:
            state = ConversationState(session_id=session_id)

        # 添加消息
        state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })

        # 更新状态
        state.current_intent = intent
        state.entities.update(entities)

        # 如果分析出偏好，更新缓存
        if analysis.get("is_preference") and analysis.get("preferences"):
            state.preferences_cache.update({
                p["category"]: p["value"]
                for p in analysis["preferences"]
            })

        # 保存状态
        self.short_term.save_conversation_state(session_id, state)

    def _generate_memory_response(self, analysis: Dict, stored_data: Dict) -> str:
        """生成记忆操作的友好响应"""
        if analysis.get("is_preference") and stored_data.get("preferences"):
            count = len(stored_data["preferences"])
            return f"已保存{count}条偏好到您的长期记忆"

        if stored_data.get("travel_history"):
            dest = stored_data["travel_history"].get("destination", "")
            return f"已记录您去{dest}的行程"

        return "记忆已更新"

    # ==================== 对外查询接口 ====================

    async def get_user_preferences(self, user_id: str, category: str = None) -> Dict:
        """
        获取用户偏好
        优先查Redis缓存，没有则查PostgreSQL并缓存
        """
        # 先查短期记忆缓存
        cached = self.short_term.get_cached_preferences(user_id)
        if cached:
            if category:
                return {category: cached.get(category)}
            return cached

        # 查长期记忆
        prefs = self.long_term.get_preferences(user_id, category)
        if prefs:
            # 缓存到Redis
            prefs_dict = {p["category"]: p["preference_value"] for p in prefs}
            self.short_term.cache_preferences(user_id, prefs_dict)
            return prefs_dict

        return {}

    async def get_travel_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取行程历史"""
        return self.long_term.get_travel_history(user_id, limit)

    async def get_conversation_context(self, session_id: str, max_turns: int = 5) -> str:
        """获取对话上下文"""
        return self.short_term.get_recent_context(session_id, max_turns)

    # ==================== Skill预留位置 ====================

    # TODO: 添加更多记忆相关Skill
    # - preference_learning_skill: 从历史行为中学习偏好
    # - memory_consolidation_skill: 记忆整合（睡眠时运行）
    # - context_aware_skill: 根据上下文自动补全信息