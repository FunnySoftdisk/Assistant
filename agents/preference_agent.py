"""
偏好查询Agent - 管理用户偏好设置
通过短期记忆缓存快速访问，长期记忆持久化存储
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
from typing import Optional, Union, List, Dict

from core.llm_client import llm_chat
from core.utils import safe_json_parse
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory


class PreferenceAgent(AgentBase):
    """
    偏好查询Agent - 管理用户偏好设置
    职责：
    1. 查询用户偏好（优先Redis缓存，其次PostgreSQL）
    2. 更新用户偏好（追加/覆盖）
    3. 缓存偏好查询结果

    数据流：
    - 短期记忆(Redis): 缓存查询结果，TTL 1小时
    - 长期记忆(PostgreSQL): 持久化存储用户偏好
    """

    SYSTEM_PROMPT = """你是一个偏好管理助手，负责理解和处理用户偏好。

偏好类型包括：
- hotel: 酒店品牌偏好（如汉庭、如家、万豪等）
- airline: 航空公司偏好（如国航、东航、南航等）
- seat: 座位偏好（如靠窗、靠过道）
- room: 房型偏好（大床房、双床房等）
- food: 餐饮偏好（中餐、西餐、小吃等）
- transport: 交通偏好（地铁、公交、打车等）
- budget: 预算等级（经济型、舒适型、高端型）
- time: 时间偏好（早起、晚睡等）

用户偏好表达模式：
1. 追加模式：使用"还"、"也"、"另外"等词，表示追加新偏好
2. 覆盖模式：使用"改成"、"变为"、"换"等词，表示替换旧偏好

输出JSON格式：
{
    "action": "append" | "update" | "query",
    "preferences": {
        "category": {"key": "value"},
        ...
    },
    "reasoning": "推理说明"
}"""

    def __init__(self, name: str = "PreferenceAgent", model_config: dict = None, **kwargs):
        super().__init__()
        self.name = name
        self.model_config = model_config or {}
        self.memory = InMemoryMemory()

        # 初始化记忆系统
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """处理偏好查询/管理请求"""
        if x is None:
            return Msg(name=self.name, content=json.dumps({"error": "No input"}), role="assistant")

        if isinstance(x, list):
            x = x[-1]

        query = x.content if hasattr(x, 'content') else str(x)

        # 判断是查询还是更新
        is_query = any(kw in query.lower() for kw in ["查询", "看看", "有什么", "list", "show", "我的偏好"])

        if is_query:
            result = await self._query_preference(query)
        else:
            result = await self._update_preference(query)

        return Msg(
            name=self.name,
            content=json.dumps(result, ensure_ascii=False),
            role="assistant"
        )

    async def _query_preference(self, query: str) -> Dict:
        """
        查询用户偏好
        1. 先查Redis缓存
        2. 缓存未命中则查PostgreSQL
        3. 结果缓存到Redis
        """
        # 提取查询类别
        category = self._extract_category(query)

        # 从短期记忆获取缓存
        user_id = "default"
        cached = self.short_term.get_cached_preferences(user_id)

        if cached and not category:
            return {
                "action": "query",
                "preferences": cached,
                "source": "cache",
                "response": f"从缓存获取到 {len(cached)} 项偏好"
            }

        # 查询长期记忆
        prefs = self.long_term.get_preferences(user_id, category)

        if not prefs:
            return {
                "action": "query",
                "preferences": {},
                "response": "您还没有设置偏好"
            }

        # 转换为dict格式
        prefs_dict = {}
        for p in prefs:
            cat = p.get("category", "general")
            if cat not in prefs_dict:
                prefs_dict[cat] = {}
            prefs_dict[cat][p.get("preference_key", "value")] = p.get("preference_value")

        # 缓存结果
        self.short_term.cache_preferences(user_id, prefs_dict)

        return {
            "action": "query",
            "preferences": prefs_dict,
            "source": "database",
            "response": self._format_preferences_response(prefs_dict)
        }

    async def _update_preference(self, query: str) -> Dict:
        """使用LLM理解并更新偏好"""
        # 构建对话上下文
        context = f"用户偏好表达: {query}"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ]

        try:
            response = await llm_chat(messages)
            result = safe_json_parse(response)

            if result is None:
                return {
                    "action": "update",
                    "error": "JSON解析失败",
                    "response": "更新偏好时出现问题"
                }

            action = result.get("action", "update")
            preferences = result.get("preferences", {})

            # 保存偏好
            user_id = "default"
            for category, prefs in preferences.items():
                if isinstance(prefs, dict):
                    for key, value in prefs.items():
                        self._save_preference(user_id, category, key, value, action)

            # 使缓存失效
            self.short_term.invalidate_preferences_cache(user_id)

            return {
                "action": "update",
                "preferences": preferences,
                "response": f"已更新您的偏好设置"
            }

        except Exception as e:
            return {
                "action": "update",
                "error": str(e),
                "response": "更新偏好时出现问题"
            }

    def _save_preference(self, user_id: str, category: str, key: str, value: str, action: str = "update") -> bool:
        """保存偏好到长期记忆"""
        from memory.long_term import UserPreference

        preference = UserPreference(
            user_id=user_id,
            category=category,
            key=key,
            value=value,
            confidence=0.9,
            source="conversation"
        )
        return self.long_term.save_preference(preference)

    def _extract_category(self, query: str) -> str:
        """从查询中提取偏好类别"""
        categories = ["hotel", "airline", "food", "transport", "budget", "seat", "room", "time"]
        query_lower = query.lower()

        for cat in categories:
            if cat in query_lower:
                return cat
        return None

    def _format_preferences_response(self, prefs: Dict) -> str:
        """格式化偏好响应"""
        if not prefs:
            return "暂无偏好记录"

        parts = []
        for category, items in prefs.items():
            if isinstance(items, dict):
                item_str = ", ".join([f"{k}:{v}" for k, v in items.items()])
                parts.append(f"{category}: {item_str}")
            else:
                parts.append(f"{category}: {items}")

        return "您的偏好：" + "；".join(parts)

    # ==================== Skill预留位置 ====================

    # TODO: preference_learning_skill - 从历史行为中学习偏好
    # TODO: preference_suggestion_skill - 基于历史推荐新偏好