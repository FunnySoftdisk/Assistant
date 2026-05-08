"""
记忆更新Agent - 智能记忆管理
使用LLM分析对话内容，提取偏好，决定是否存储到长期记忆

架构：
- match_preferences(query): 从记忆库中匹配与查询相关的偏好
- summarize_preferences(prefs): 将匹配到的偏好汇总成可用格式
- analyze_and_store(): 分析对话，决定是否更新记忆
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
import asyncio
import re
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time

from core.llm_client import llm_chat
from core.utils import safe_json_parse
from memory.short_term import ShortTermMemory, ConversationState
from memory.long_term import (
    LongTermMemory, UserPreference, TravelHistory,
    PreferenceCategory, PreferenceSource, ConfidenceCalculator,
    PreferenceConflictResolver, StorageManager, PreferenceExtractor
)


@dataclass
class MemoryResult:
    """记忆操作结果"""
    action: str  # "match", "store", "summarize", "skip"
    success: bool
    matched_preferences: List[Dict] = field(default_factory=list)
    stored_data: Optional[Dict] = None
    summarized_prefs: Dict = None
    response: str = ""
    confidence: float = 0.0


class MemoryAgent(AgentBase):
    """
    记忆更新Agent - 智能记忆管理

    核心职责：
    1. 偏好匹配 (match_preferences)
       - 根据用户查询，从记忆库中检索相关偏好
       - 使用标签、关键词、上下文多维度匹配

    2. 偏好汇总 (summarize_preferences)
       - 将匹配到的偏好转换为规划Agent可用的格式
       - 处理冲突、优先级、时间衰减

    3. 智能存储 (analyze_and_store)
       - 判断是否需要存储新偏好
       - 处理偏好冲突
       - 管理存储生命周期

    4. 对外查询接口
       - get_user_preferences()
       - get_travel_history()
       - get_conversation_context()

    特性:
    - LLM调用超时保护（30秒）
    - 异常时使用关键词回退
    - 数据库连接失败时优雅降级
    - 存储容量管理（归档/压缩/清理）
    """

    SYSTEM_PROMPT = """你是一个记忆管理助手，负责分析对话内容，提取和管理用户偏好。

## 你的职责

### 1. 判断是否需要存储
用户是否在表达新的偏好信息？包括：
- 明确偏好："我爱吃辣"、"我喜欢汉庭"
- 习惯表达："我一般早上9点打球"
- 否定偏好："我不要香菜"、"不住如家"
- 正反馈："好的"、"可以"、"就用这个"

### 2. 提取偏好信息
从对话中提取：
- preference_type: habit(习惯) | negative(否定) | situational(场景) | reminder(提醒) | relational(关系)
- description: 自然语言描述
- conditions: 触发条件数组
- tags: 标签数组，用于检索
- confidence: 置信度 (0-1)

### 3. 判断更新策略
- 新偏好 vs 已有偏好 → 追加还是更新？
- 冲突检测 → 是否与现有偏好矛盾？
- 时间权重 → 新的是否更可信？

### 4. 输出格式

分析结果JSON：
{
    "should_store": true/false,
    "preferences": [
        {
            "preference_type": "habit|negative|situational|reminder",
            "description": "自然语言描述",
            "conditions": [{"weather": "晴天"}, {"time": "09:00"}],
            "tags": ["辣", "川菜"],
            "behavior": "偏好辣的食物",
            "confidence": 0.85,
            "is_explicit": true,
            "source": "explicit|implicit|behavior",
            "reasoning": "用户明确表达了..."
        }
    ],
    "update_strategy": "append|update|skip",
    "conflict_info": "如果有冲突，说明情况",
    "reasoning": "判断理由"
}

## 偏好类型说明

| 类型 | 例子 | 处理方式 |
|------|------|----------|
| habit | "我爱吃辣" | 追加到同类偏好 |
| negative | "不要香菜" | 优先级最高，不可覆盖 |
| situational | "出差住酒店要有接送" | 关联场景标签 |
| reminder | "9点要吃药" | 创建定时提醒 |
| relational | "我爱吃辣但老公不爱" | 分离存储，关联用户 |

## 置信度来源

- 显式表达（我要/我爱）：0.85-0.95
- 隐式表达（一般/通常）：0.6-0.75
- 行为观察（点了火锅）：0.5-0.7
- 单次出现：基础0.5
- 多次确认：+0.1 per occurrence, max 0.95"""

    def __init__(self, name: str = "MemoryAgent", model_config: dict = None, **kwargs):
        super().__init__()
        self.name = name
        self.model_config = model_config or {}

        # 初始化记忆系统
        self.short_term = ShortTermMemory()  # Redis存储
        self.long_term = LongTermMemory()    # PostgreSQL存储

        # 初始化各组件
        self.extractor = PreferenceExtractor()
        self.storage_manager = None
        if self.long_term._pool:
            self.storage_manager = StorageManager(self.long_term)

        # 尝试初始化数据库
        try:
            self.long_term.init_database()
        except Exception as e:
            print(f"⚠️ MemoryAgent数据库初始化失败: {e}")

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """处理记忆请求"""
        if x is None:
            return Msg(name=self.name, content=json.dumps({"error": "No input"}), role="assistant")

        if isinstance(x, list):
            x = x[-1]

        try:
            data = json.loads(x.content) if isinstance(x.content, str) else x.content
        except:
            data = {"content": str(x.content)}

        action = data.get("action", "analyze")

        try:
            if action == "match":
                result = await self.match_preferences(data)
            elif action == "summarize":
                result = await self.summarize_preferences(data)
            elif action == "analyze":
                result = await self.analyze_and_store(data)
            elif action == "query":
                result = await self.get_user_preferences(data)
            else:
                result = await self.analyze_and_store(data)
        except asyncio.TimeoutError:
            result = MemoryResult(
                action=action,
                success=False,
                response="记忆处理超时"
            )
        except Exception as e:
            result = MemoryResult(
                action=action,
                success=False,
                response=f"记忆处理失败: {str(e)[:50]}"
            )

        # 转换为可序列化字典
        result_dict = {
            "action": result.action,
            "success": result.success,
            "matched_preferences": result.matched_preferences,
            "stored_data": result.stored_data,
            "summarized_prefs": result.summarized_prefs,
            "response": result.response,
            "confidence": result.confidence
        }

        return Msg(
            name=self.name,
            content=json.dumps(result_dict, ensure_ascii=False),
            role="assistant"
        )

    async def match_preferences(self, data: dict) -> MemoryResult:
        """
        匹配相关偏好

        输入: {"query": "帮我规划去上海的行程", "user_id": "default"}
        输出: 匹配到的偏好列表
        """
        query = data.get("query", "")
        user_id = data.get("user_id", "default")
        entities = data.get("entities", {})

        # 1. 提取查询关键词和实体
        query_tags = self._extract_query_tags(query, entities)

        # 2. 从数据库检索相关偏好
        matched = await self._find_matching_preferences(user_id, query_tags)

        # 3. 过滤和排序
        filtered = self._filter_by_relevance(matched, query_tags)

        # 4. 计算每个偏好的当前有效置信度
        for pref in filtered:
            pref["effective_confidence"] = self._calculate_effective_confidence(pref)

        # 按置信度排序
        filtered.sort(key=lambda x: x.get("effective_confidence", 0), reverse=True)

        return MemoryResult(
            action="match",
            success=True,
            matched_preferences=filtered,
            response=f"匹配到 {len(filtered)} 条相关偏好"
        )

    async def summarize_preferences(self, data: dict) -> MemoryResult:
        """
        汇总偏好供规划使用

        输入: {"preferences": [...], "context": "上海旅行"}
        输出: 汇总后的偏好字典，按类别组织
        """
        preferences = data.get("preferences", [])
        context = data.get("context", "")

        if not preferences:
            return MemoryResult(
                action="summarize",
                success=True,
                summarized_prefs={},
                response="无相关偏好"
            )

        # 按类别分组
        summarized = {
            "hotel": [],
            "transport": [],
            "food": [],
            "sports": [],
            "time": [],
            "weather": [],
            "reminders": [],
            "negatives": [],
            "situational": []
        }

        for pref in preferences:
            category = pref.get("category", "general")
            pref_type = pref.get("preference_type", "habit")

            summary_item = {
                "description": pref.get("description", ""),
                "value": pref.get("preference_value", ""),
                "confidence": pref.get("effective_confidence", pref.get("confidence", 0.5)),
                "tags": pref.get("tags", []),
                "conditions": pref.get("conditions", [])
            }

            if pref_type == "negative":
                summarized["negatives"].append(summary_item)
            elif pref_type == "reminder":
                summarized["reminders"].append(summary_item)
            elif pref_type == "situational":
                summarized["situational"].append(summary_item)
            elif category in summarized:
                summarized[category].append(summary_item)
            else:
                summarized.setdefault(category, []).append(summary_item)

        # 清理空类别
        summarized = {k: v for k, v in summarized.items() if v}

        return MemoryResult(
            action="summarize",
            success=True,
            summarized_prefs=summarized,
            response="偏好汇总完成"
        )

    async def analyze_and_store(self, data: dict) -> MemoryResult:
        """
        分析对话内容，决定是否存储

        输入: {"content": "用户说的话", "session_id": "...", "user_id": "..."}
        输出: 存储结果
        """
        user_input = data.get("content", "")
        session_id = data.get("session_id", "default")
        user_id = data.get("user_id", "default")
        intent = data.get("intent", "")
        entities = data.get("entities", {})

        # 1. 使用LLM分析是否需要存储
        analysis = await self._analyze_for_storage(user_input, intent)

        stored_data = {}
        new_prefs = []

        # 2. 如果应该存储，提取并保存偏好
        if analysis.get("should_store") and analysis.get("preferences"):
            for pref_data in analysis["preferences"]:
                try:
                    success, pref = await self._store_single_preference(user_id, pref_data)
                    if success:
                        new_prefs.append(pref)
                        stored_data[f"pref_{len(new_prefs)}"] = pref_data
                except Exception as e:
                    print(f"⚠️ 存储偏好失败: {e}")

        # 3. 存储行程历史（如果是旅行规划）
        if intent == "travel_planning" and entities.get("locations"):
            try:
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
            except Exception as e:
                print(f"⚠️ 存储行程失败: {e}")

        # 4. 更新短期记忆
        try:
            await self._update_short_term(session_id, user_id, user_input, intent, entities, analysis)
        except Exception:
            pass

        # 5. 生成响应
        if new_prefs:
            response = f"已保存 {len(new_prefs)} 条偏好到记忆"
        elif stored_data.get("travel_history"):
            response = f"已记录去 {stored_data['travel_history']['destination']} 的行程"
        else:
            response = analysis.get("reasoning", "无需更新记忆")

        return MemoryResult(
            action="store",
            success=True,
            stored_data=stored_data,
            response=response,
            confidence=analysis.get("confidence", 0.5)
        )

    async def _analyze_for_storage(self, user_input: str, intent: str) -> Dict:
        """使用LLM分析是否需要存储"""
        # 简单意图直接跳过
        if intent in ["info_query", "greeting", "general_chat"]:
            return {
                "should_store": False,
                "reasoning": f"意图 {intent} 不需要存储",
                "preferences": []
            }

        # 构建分析消息
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"分析以下对话，判断是否需要存储偏好：\n\n用户输入: {user_input}\n当前意图: {intent}"}
        ]

        try:
            response = await asyncio.wait_for(llm_chat(messages), timeout=30.0)
            result = safe_json_parse(response)
            if result:
                return result
        except Exception as e:
            print(f"⚠️ LLM分析失败，使用关键词回退: {e}")

        # 回退到关键词分析
        return self._keyword_fallback_analysis(user_input)

    def _keyword_fallback_analysis(self, text: str) -> Dict:
        """基于关键词的回退分析"""
        prefs = self.extractor.extract_from_text(text)

        if not prefs:
            return {
                "should_store": False,
                "reasoning": "未检测到偏好表达",
                "preferences": []
            }

        return {
            "should_store": True,
            "reasoning": "关键词检测到偏好",
            "preferences": prefs,
            "confidence": 0.5
        }

    async def _store_single_preference(self, user_id: str, pref_data: Dict) -> Tuple[bool, UserPreference]:
        """存储单个偏好"""
        # 创建偏好对象
        preference = UserPreference(
            user_id=user_id,
            category=pref_data.get("category", PreferenceCategory.GENERAL.value),
            key=pref_data.get("key", "habit"),
            value=pref_data.get("value") or pref_data.get("description", ""),
            confidence=pref_data.get("confidence", 0.6),
            source=pref_data.get("source", "conversation"),
            is_explicit=pref_data.get("is_explicit", False),
            occurrence_count=1,
            metadata={
                "description": pref_data.get("description", ""),
                "conditions": pref_data.get("conditions", []),
                "tags": pref_data.get("tags", []),
                "preference_type": pref_data.get("preference_type", "habit")
            }
        )

        # 保存并获取结果
        result = self.long_term.save_preference(preference)

        if isinstance(result, dict) and result.get("success"):
            return True, preference
        return False, preference

    def _extract_query_tags(self, query: str, entities: Dict) -> List[str]:
        """从查询中提取标签"""
        tags = []
        query_lower = query.lower()

        # 目的地标签
        if entities.get("locations"):
            tags.extend(entities["locations"])

        # 时间标签
        if entities.get("date"):
            tags.append(entities["date"])

        # 季节/天气相关
        weather_keywords = ["晴天", "雨天", "夏天", "冬天", "下雨", "下雪"]
        for kw in weather_keywords:
            if kw in query_lower:
                tags.append(kw)

        # 活动相关
        activity_keywords = ["篮球", "足球", "跑步", "旅游", "出差", "打球"]
        for kw in activity_keywords:
            if kw in query_lower:
                tags.append(kw)

        # 食物相关
        food_keywords = ["辣", "火锅", "川菜", "粤菜", "早茶", "海鲜"]
        for kw in food_keywords:
            if kw in query_lower:
                tags.append(kw)

        # 交通相关
        transport_keywords = ["高铁", "飞机", "地铁", "打车"]
        for kw in transport_keywords:
            if kw in query_lower:
                tags.append(kw)

        return tags if tags else [query]

    async def _find_matching_preferences(self, user_id: str, tags: List[str]) -> List[Dict]:
        """根据标签查找匹配偏好"""
        try:
            # 获取用户所有活跃偏好
            all_prefs = self.long_term.get_preferences(user_id)

            if not all_prefs:
                return []

            matched = []

            # 逐个检查是否匹配
            for pref in all_prefs:
                pref_tags = pref.get("tags", []) or []
                metadata = pref.get("metadata", {}) or {}

                # 检查标签匹配
                if isinstance(pref_tags, list):
                    for tag in tags:
                        if tag in pref_tags:
                            pref["match_reason"] = f"标签匹配: {tag}"
                            matched.append(pref)
                            break

                # 检查条件匹配
                conditions = metadata.get("conditions", [])
                if isinstance(conditions, list):
                    for condition in conditions:
                        for tag in tags:
                            if tag in str(condition):
                                pref["match_reason"] = f"条件匹配: {condition}"
                                matched.append(pref)
                                break

            return matched

        except Exception as e:
            print(f"⚠️ 匹配偏好失败: {e}")
            return []

    def _filter_by_relevance(self, preferences: List[Dict], query_tags: List[str]) -> List[Dict]:
        """过滤相关度低的偏好"""
        if not query_tags:
            return preferences

        filtered = []
        for pref in preferences:
            # 检查是否有明确的匹配原因
            match_reason = pref.get("match_reason", "")

            # 检查标签或条件是否相关
            metadata = pref.get("metadata", {}) or {}
            conditions = metadata.get("conditions", [])
            tags = metadata.get("tags", [])

            # 至少有一个标签或条件与查询相关
            is_relevant = False
            for tag in query_tags:
                if tag in tags or tag in str(conditions):
                    is_relevant = True
                    break

            if is_relevant or match_reason:
                filtered.append(pref)

        return filtered if filtered else preferences[:10]  # 默认返回前10条

    def _calculate_effective_confidence(self, pref: Dict) -> float:
        """计算有效置信度（考虑时间衰减）"""
        base_confidence = pref.get("confidence", 0.5)
        last_updated = pref.get("last_updated")

        if not last_updated:
            return base_confidence

        # 转换为datetime
        if isinstance(last_updated, str):
            try:
                last_updated = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            except:
                return base_confidence

        # 计算时间衰减
        try:
            decay = ConfidenceCalculator._calculate_time_decay(last_updated)
            return base_confidence * decay
        except:
            return base_confidence

    async def get_user_preferences(self, data: dict) -> MemoryResult:
        """获取用户偏好"""
        user_id = data.get("user_id", "default")
        category = data.get("category")

        try:
            prefs = self.long_term.get_preferences(user_id, category)
            return MemoryResult(
                action="query",
                success=True,
                matched_preferences=prefs,
                response=f"获取到 {len(prefs)} 条偏好"
            )
        except Exception as e:
            return MemoryResult(
                action="query",
                success=False,
                response=f"获取偏好失败: {str(e)[:50]}"
            )

    async def get_travel_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取行程历史"""
        try:
            return self.long_term.get_travel_history(user_id, limit)
        except Exception:
            return []

    async def get_conversation_context(self, session_id: str, max_turns: int = 5) -> str:
        """获取对话上下文"""
        try:
            return self.short_term.get_recent_context(session_id, max_turns)
        except Exception:
            return ""

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
        try:
            state = self.short_term.get_conversation_state(session_id)
            if state is None:
                state = ConversationState(session_id=session_id)

            state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time()
            })

            state.current_intent = intent
            state.entities.update(entities)

            if analysis.get("preferences"):
                state.preferences_cache.update({
                    p.get("category", "general"): p.get("value", "")
                    for p in analysis["preferences"]
                })

            self.short_term.save_conversation_state(session_id, state)
        except Exception as e:
            print(f"⚠️ 短期记忆更新失败: {e}")

    # ==================== 偏好提取器 ====================

    @dataclass
    class ExtractedPreference:
        category: str
        key: str
        value: str
        description: str
        preference_type: str
        tags: List[str]
        conditions: List[Dict]
        confidence: float
        is_explicit: bool
        source: str


class PreferenceExtractor:
    """从文本中提取偏好信息"""

    # 偏好类别关键词映射
    CATEGORY_KEYWORDS = {
        "food": ["吃", "美食", "餐厅", "菜", "辣", "火锅", "川菜", "粤菜", "早茶", "早餐", "午餐", "晚餐", "香菜"],
        "hotel": ["酒店", "住宿", "住", "汉庭", "如家", "万豪", "希尔顿", "民宿", "接送"],
        "transport": ["高铁", "飞机", "火车", "地铁", "公交", "打车", "开车", "自驾"],
        "sports": ["篮球", "足球", "跑步", "游泳", "健身", "运动", "打球", "羽毛球"],
        "time": ["早上", "上午", "中午", "下午", "晚上", "几点", "时间", "9点", "9点钟"],
        "weather": ["晴天", "雨天", "阴天", "下雨", "晴天", "天气"],
        "location": ["北京", "上海", "杭州", "旅游", "出差", "去", "到"],
    }

    @classmethod
    def extract_from_text(cls, text: str) -> List[Dict]:
        """从文本中提取偏好"""
        results = []
        text_lower = text.lower()

        # 检测显式偏好
        explicit_patterns = [
            (r"我(爱|喜欢|想要|要|prefer)\s*(.+)", "explicit"),
            (r"(从来|绝对|一定|必须)\s*(不)?(喜欢|吃|住|坐|用)", "explicit"),
            (r"不要|不爱|不喜欢|不想|别", "negative"),
        ]

        for pattern, pref_type in explicit_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                pref = cls._parse_match(match, pref_type, text)
                if pref:
                    results.append(pref)

        # 基于关键词推断类别
        for pref in results:
            if pref.get("category") == "general":
                pref["category"] = cls._infer_category(text_lower)

        # 提取复合条件偏好
        compound_prefs = cls._extract_compound_conditions(text)
        results.extend(compound_prefs)

        # 提取定时提醒
        reminders = cls._extract_reminders(text)
        results.extend(reminders)

        return results

    @classmethod
    def _parse_match(cls, match: re.Match, pref_type: str, text: str) -> Optional[Dict]:
        """解析匹配结果"""
        try:
            full_match = match.group(0)
            value = match.group(2) if len(match.groups()) > 1 else match.group(0)

            is_negative = "不" in full_match and pref_type != "negative"

            return {
                "category": cls._infer_category(text.lower()),
                "key": "habit",
                "value": value.strip(),
                "description": full_match,
                "preference_type": "negative" if is_negative else "habit",
                "tags": cls._extract_tags(text),
                "conditions": [{"operator": "always"}],
                "confidence": 0.85 if pref_type == "explicit" else 0.6,
                "is_explicit": pref_type == "explicit",
                "source": pref_type
            }
        except Exception:
            return None

    @classmethod
    def _infer_category(cls, text: str) -> str:
        """推断类别"""
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return category
        return "general"

    @classmethod
    def _extract_tags(cls, text: str) -> List[str]:
        """提取标签"""
        tags = []
        text_lower = text.lower()
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    tags.append(kw)
        return tags

    @classmethod
    def _extract_compound_conditions(cls, text: str) -> List[Dict]:
        """提取复合条件偏好"""
        results = []

        # 晴天上午9点打篮球
        compound_pattern = r"(晴天|雨天|阴天)?(上午|下午|早上|中午|晚上)?(\d+)?点?\s*(打篮球|踢球|跑步|游泳)?"
        match = re.search(compound_pattern, text)
        if match and any(kw in text for kw in ["打篮球", "踢球", "跑步", "游泳"]):
            conditions = []
            if match.group(1):
                conditions.append({"weather": match.group(1)})
            if match.group(2):
                conditions.append({"time_of_day": match.group(2)})
            if match.group(3):
                conditions.append({"hour": match.group(3)})

            results.append({
                "category": "sports",
                "key": "activity_with_conditions",
                "value": match.group(4) or "运动",
                "description": text,
                "preference_type": "situational",
                "tags": cls._extract_tags(text),
                "conditions": conditions,
                "confidence": 0.75,
                "is_explicit": False,
                "source": "implicit"
            })

        return results

    @classmethod
    def _extract_reminders(cls, text: str) -> List[Dict]:
        """提取定时提醒"""
        results = []

        # 9点要吃药、9点钟吃药
        reminder_pattern = r"(\d+)[点时](?:钟)?\s*(要|记得|必须)?\s*(.+)"
        match = re.search(reminder_pattern, text)
        if match and any(kw in text for kw in ["吃药", "服药", "检查", "测量"]):
            results.append({
                "category": "time",
                "key": "reminder",
                "value": match.group(3),
                "description": text,
                "preference_type": "reminder",
                "tags": ["提醒", "定时", match.group(3)],
                "conditions": [{"hour": match.group(1), "recurrence": "daily"}],
                "confidence": 0.9,
                "is_explicit": True,
                "source": "explicit"
            })

        return results
