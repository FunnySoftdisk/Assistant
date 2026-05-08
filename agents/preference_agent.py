"""
偏好查询Agent - 管理用户偏好设置
通过短期记忆缓存快速访问，长期记忆持久化存储
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
import asyncio
from typing import Optional, Union, List, Dict

from core.llm_client import llm_chat
from core.utils import safe_json_parse, validate_message
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory


class PreferenceAgent(AgentBase):
    """
    偏好查询Agent - 管理用户偏好设置
    职责：
    1. 查询用户偏好（优先Redis缓存，其次PostgreSQL）
    2. 更新用户偏好（追加/覆盖）
    3. 缓存偏好查询结果

    特性:
    - 数据库连接失败时自动降级到文件存储
    - 输入验证（长度限制、敏感字符）
    - 查询结果缓存
    """

    SYSTEM_PROMPT = """你是一个偏好管理助手，负责理解和处理用户偏好。

## 你的职责

1. **理解用户偏好表达**：识别用户提到的酒店品牌、交通方式、餐饮喜好等
2. **判断操作类型**：是查询现有偏好，还是添加/更新偏好
3. **提取偏好结构**：将自然语言偏好转换为结构化数据

## 偏好类型

| 类别 | 例子 |
|------|------|
| hotel | 酒店品牌（汉庭、如家、万豪等） |
| airline | 航空公司（国航、东航、南航等） |
| seat | 座位偏好（靠窗、靠过道） |
| room | 房型（大床房，双床房等） |
| food | 餐饮（中餐、火锅、海鲜等） |
| transport | 交通（地铁、打车、公交） |
| budget | 预算（经济型、舒适型、高端型） |

## 用户表达模式

- **追加**："我喜欢汉庭"、"还喜欢万豪" → action: "append"
- **覆盖**："换成如家"、"改成经济型" → action: "update"

## 输出格式

当需要更新偏好时：
{
    "action": "append" | "update",
    "preferences": {
        "hotel": {"brand": "汉庭"},
        "food": {"cuisine": "火锅"}
    },
    "reasoning": "用户表达了酒店和餐饮偏好"
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

        # 验证输入
        is_valid, error_msg = validate_message(query, max_length=1000)
        if not is_valid:
            return Msg(
                name=self.name,
                content=json.dumps({
                    "action": "error",
                    "error": error_msg,
                    "response": f"输入验证失败: {error_msg}"
                }, ensure_ascii=False),
                role="assistant"
            )

        # 判断是查询还是更新
        is_query = any(kw in query.lower() for kw in ["查询", "看看", "有什么", "list", "show", "我的偏好"])

        try:
            if is_query:
                result = await self._query_preference(query)
            else:
                result = await self._update_preference(query)
        except asyncio.TimeoutError:
            result = {
                "action": "error",
                "error": "LLM调用超时",
                "response": "处理超时，请稍后重试"
            }
        except Exception as e:
            result = {
                "action": "error",
                "error": str(e),
                "response": f"处理失败: {str(e)[:50]}"
            }

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

        try:
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
            try:
                self.short_term.cache_preferences(user_id, prefs_dict)
            except Exception:
                pass

            return {
                "action": "query",
                "preferences": prefs_dict,
                "source": "database",
                "response": self._format_preferences_response(prefs_dict)
            }
        except Exception as e:
            return {
                "action": "query",
                "preferences": {},
                "error": str(e),
                "response": f"查询偏好时出现问题: {str(e)[:50]}"
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
            response = await asyncio.wait_for(llm_chat(messages), timeout=30.0)
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
                        try:
                            self._save_preference(user_id, category, key, value, action)
                        except Exception as e:
                            pass  # 单个保存失败不影响其他

            # 使缓存失效
            try:
                self.short_term.invalidate_preferences_cache(user_id)
            except Exception:
                pass

            return {
                "action": "update",
                "preferences": preferences,
                "response": f"已更新您的偏好设置"
            }

        except asyncio.TimeoutError:
            return {
                "action": "update",
                "error": "LLM调用超时",
                "response": "更新偏好超时，请稍后重试"
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

        try:
            preference = UserPreference(
                user_id=user_id,
                category=category,
                key=key,
                value=value,
                confidence=0.9,
                source="conversation"
            )
            return self.long_term.save_preference(preference)
        except Exception as e:
            return False

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