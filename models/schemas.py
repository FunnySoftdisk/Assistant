"""
数据模型定义
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum


class IntentType(str, Enum):
    """意图类型枚举"""
    TRAVEL_PLANNING = "travel_planning"
    MEMORY_QUERY = "memory_query"
    PREFERENCE_MANAGE = "preference_manage"
    INFO_QUERY = "info_query"
    EVENT_COLLECTION = "event_collection"
    EXECUTION = "execution"
    UNKNOWN = "unknown"


class Message(BaseModel):
    """对话消息"""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: Optional[float] = None


class IntentResult(BaseModel):
    """意图识别结果"""
    intent: IntentType
    confidence: float
    entities: Dict[str, Any] = {}
    requires_skills: List[str] = []


class TaskResult(BaseModel):
    """任务执行结果"""
    agent: str
    success: bool
    data: Any
    error: Optional[str] = None


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    user_id: Optional[str] = "default"
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    session_id: str
    tasks_executed: List[str] = []
    memory_updated: bool = False