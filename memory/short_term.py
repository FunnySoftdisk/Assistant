"""
短期记忆模块 - Redis存储
"""
import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class ConversationState:
    """对话状态"""
    session_id: str
    messages: List[Dict] = field(default_factory=list)
    current_intent: str = ""
    entities: Dict[str, Any] = field(default_factory=dict)
    preferences_cache: Dict[str, Any] = field(default_factory=dict)
    recent_agents: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class ShortTermMemory:
    """
    短期记忆 - Redis存储
    支持：
    1. 对话状态（Session级）
    2. PreferenceAgent查询结果的缓存
    3. 最近对话上下文
    """

    def __init__(
        self,
        redis_config: dict = None,
        ttl: int = 3600,
        max_history: int = 20
    ):
        self.ttl = ttl
        self.max_history = max_history

        # Redis配置
        self.redis_config = redis_config or {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": None,
            "decode_responses": True
        }
        self._redis = None

        # 本地后备存储（当Redis不可用时）
        self._local_store: Dict[str, Dict] = {}

        # 尝试连接Redis
        if REDIS_AVAILABLE:
            try:
                self._redis = redis.Redis(**self.redis_config)
                self._redis.ping()
                print("✓ Redis连接成功")
            except Exception as e:
                print(f"⚠️ Redis连接失败: {e}，使用本地存储")
                self._redis = None

    # ==================== 对话状态管理 ====================

    def save_conversation_state(self, session_id: str, state: ConversationState) -> bool:
        """保存对话状态"""
        key = f"conv_state:{session_id}"
        data = {
            "session_id": state.session_id,
            "messages": state.messages[-self.max_history:],  # 只保留最近N条
            "current_intent": state.current_intent,
            "entities": state.entities,
            "preferences_cache": state.preferences_cache,
            "recent_agents": state.recent_agents[-10:],  # 最近10个Agent
            "created_at": state.created_at,
            "last_active": time.time()
        }

        if self._redis:
            try:
                self._redis.setex(key, self.ttl, json.dumps(data))
                return True
            except Exception as e:
                print(f"⚠️ Redis存储失败: {e}")

        # 本地后备
        self._local_store[key] = data
        return True

    def get_conversation_state(self, session_id: str) -> Optional[ConversationState]:
        """获取对话状态"""
        key = f"conv_state:{session_id}"

        if self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    state_data = json.loads(data)
                    return ConversationState(**state_data)
            except Exception as e:
                print(f"⚠️ Redis读取失败: {e}")

        # 本地后备
        data = self._local_store.get(key)
        if data:
            return ConversationState(**data)
        return None

    def update_conversation_state(self, session_id: str, **kwargs) -> bool:
        """更新对话状态的特定字段"""
        state = self.get_conversation_state(session_id)
        if state is None:
            state = ConversationState(session_id=session_id)

        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)

        state.last_active = time.time()
        return self.save_conversation_state(session_id, state)

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """添加消息到对话历史"""
        state = self.get_conversation_state(session_id)
        if state is None:
            state = ConversationState(session_id=session_id)

        state.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        state.last_active = time.time()

        return self.save_conversation_state(session_id, state)

    # ==================== 偏好缓存管理 ====================

    def cache_preferences(self, user_id: str, preferences: Dict[str, Any], ttl: int = None) -> bool:
        """缓存用户偏好（从长期记忆查询的结果）"""
        key = f"prefs_cache:{user_id}"
        cache_data = {
            "preferences": preferences,
            "cached_at": time.time(),
            "expires_at": time.time() + (ttl or self.ttl)
        }

        if self._redis:
            try:
                self._redis.setex(key, ttl or self.ttl, json.dumps(cache_data))
                return True
            except Exception as e:
                print(f"⚠️ 偏好缓存失败: {e}")

        self._local_store[key] = cache_data
        return True

    def get_cached_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取缓存的偏好（用于PreferenceAgent快速查询）"""
        key = f"prefs_cache:{user_id}"

        if self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    cache_data = json.loads(data)
                    # 检查是否过期
                    if cache_data.get("expires_at", 0) > time.time():
                        return cache_data.get("preferences")
                    else:
                        # 已过期，删除
                        self._redis.delete(key)
            except Exception as e:
                print(f"⚠️ 偏好缓存读取失败: {e}")

        # 本地后备
        cache_data = self._local_store.get(key)
        if cache_data and cache_data.get("expires_at", 0) > time.time():
            return cache_data.get("preferences")
        return None

    def invalidate_preferences_cache(self, user_id: str) -> bool:
        """使偏好缓存失效（当长期记忆更新时调用）"""
        key = f"prefs_cache:{user_id}"

        if self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass

        if key in self._local_store:
            del self._local_store[key]
        return True

    # ==================== Agent执行追踪 ====================

    def record_agent_execution(self, session_id: str, agent_name: str, result: Any) -> bool:
        """记录Agent执行历史"""
        key = f"agent_exec:{session_id}"

        if self._redis:
            try:
                history = self._redis.lrange(key, 0, -1) or []
                history.append(json.dumps({
                    "agent": agent_name,
                    "result": str(result)[:200],  # 截断
                    "timestamp": time.time()
                }))
                # 只保留最近50条
                if len(history) > 50:
                    history = history[-50:]
                self._redis.delete(key)
                for item in history:
                    self._redis.rpush(key, item)
                self._redis.expire(key, self.ttl)
                return True
            except Exception as e:
                print(f"⚠️ Agent执行记录失败: {e}")

        return True

    # ==================== 工具方法 ====================

    def get_recent_context(self, session_id: str, max_turns: int = 5) -> str:
        """获取最近N轮对话的上下文字符串"""
        state = self.get_conversation_state(session_id)
        if not state or not state.messages:
            return ""

        messages = state.messages[-max_turns * 2:]  # 用户+助手=2条/轮
        context_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]  # 截断
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def clear_session(self, session_id: str) -> bool:
        """清空会话数据"""
        keys_to_delete = [
            f"conv_state:{session_id}",
            f"prefs_cache:{session_id}",
            f"agent_exec:{session_id}"
        ]

        if self._redis:
            try:
                for key in keys_to_delete:
                    self._redis.delete(key)
            except Exception:
                pass

        for key in keys_to_delete:
            if key in self._local_store:
                del self._local_store[key]
        return True

    def cleanup_expired(self) -> int:
        """清理过期数据（本地存储用）"""
        current_time = time.time()
        cleaned = 0

        for key in list(self._local_store.keys()):
            if key.startswith("prefs_cache:"):
                data = self._local_store[key]
                if data.get("expires_at", 0) < current_time:
                    del self._local_store[key]
                    cleaned += 1

        return cleaned