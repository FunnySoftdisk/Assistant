"""
FastAPI路由定义
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from models.schemas import ChatRequest, ChatResponse, IntentType
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import uuid
import json
import time
import asyncio
from collections import defaultdict

# 导入Agent
from agents.intention_agent import IntentionAgent
from agents.orchestration_agent import OrchestrationAgent
from agents.preference_agent import PreferenceAgent
from agents.info_query_agent import InfoQueryAgent
from agents.execution_agent import ExecutionAgent
from agents.planning_agent import PlanningAgent
from agents.summarization_agent import SummarizationAgent
from agents.memory_agent import MemoryAgent
from agentscope.message import Msg

# 导入Skill系统
from skills.skill_registry import get_skill_registry, load_skill

# 导入工具函数
from core.utils import validate_message, truncate_text

# 创建路由
router = APIRouter(prefix="/api")

# 全局Agent实例
agents: Dict = {}
_skill_registry = None

# ============================================================
# 对话历史管理器
# ============================================================
class ConversationHistoryManager:
    """
    对话历史管理器
    负责：
    1. 维护会话消息历史
    2. 提供上下文给意图识别
    3. 管理历史消息截断
    """

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self._history: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def add_message(self, session_id: str, role: str, content: str, metadata: dict = None):
        """添加消息到历史"""
        async with self._lock:
            message = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "metadata": metadata or {}
            }
            self._history[session_id].append(message)

            # 截断过长的历史
            if len(self._history[session_id]) > self.max_history * 2:
                # 保留最近的消息和早期摘要
                self._history[session_id] = self._history[session_id][-self.max_history:]

    async def get_context(self, session_id: str, max_turns: int = 5) -> str:
        """
        获取对话上下文字符串
        用于传给 LLM 进行意图识别
        """
        async with self._lock:
            history = self._history.get(session_id, [])

        if not history:
            return ""

        # 获取最近 N 轮（每轮2条消息：用户+助手）
        recent = history[-max_turns * 2:] if len(history) > max_turns * 2 else history

        context_parts = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # 截断过长的消息
            content = truncate_text(content, 200)
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    async def get_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """获取会话历史"""
        async with self._lock:
            history = self._history.get(session_id, [])
            return history[-limit:] if len(history) > limit else history

    async def clear_history(self, session_id: str):
        """清空会话历史"""
        async with self._lock:
            if session_id in self._history:
                del self._history[session_id]

    async def get_turn_count(self, session_id: str) -> int:
        """获取对话轮数"""
        async with self._lock:
            history = self._history.get(session_id, [])
            # 每2条消息算一轮
            return len(history) // 2


# 全局对话历史管理器
_history_manager = ConversationHistoryManager(max_history=20)

# ============================================================
# 限流配置
# ============================================================
class RateLimiter:
    """简单内存限流器"""

    def __init__(self):
        self._requests = defaultdict(list)  # ip -> [(timestamp, count)]
        self._daily_counts = defaultdict(int)  # ip -> daily count
        self._last_reset = datetime.now().date()

    def _cleanup(self):
        """清理过期记录"""
        now = datetime.now().date()
        if now != self._last_reset:
            self._daily_counts.clear()
            self._last_reset = now

        cutoff = time.time() - 3600  # 1小时前的请求
        for ip in self._requests:
            self._requests[ip] = [
                (t, c) for t, c in self._requests[ip]
                if t > cutoff
            ]

    def is_allowed(self, key: str, max_per_minute: int = 60, max_per_day: int = 1000) -> tuple:
        """
        检查是否允许请求
        Returns: (allowed, remaining_minute, remaining_day, error_msg)
        """
        self._cleanup()

        now = time.time()
        minute_key = f"{key}:minute"

        # 检查每分钟限制
        recent_requests = [
            t for t, _ in self._requests.get(key, [])
            if now - t < 60
        ]

        if len(recent_requests) >= max_per_minute:
            return False, 0, max_per_day - self._daily_counts.get(key, 0), "请求过于频繁，请稍后重试"

        # 检查每日限制
        if self._daily_counts.get(key, 0) >= max_per_day:
            return False, max_per_minute - len(recent_requests), 0, "今日请求次数已达上限"

        # 记录请求
        self._requests[key].append((now, 1))
        self._daily_counts[key] = self._daily_counts.get(key, 0) + 1

        return True, max_per_minute - len(recent_requests) - 1, max_per_day - self._daily_counts[key] - 1, ""


_rate_limiter = RateLimiter()

# ============================================================
# Agent初始化
# ============================================================

def init_agents():
    """初始化所有Agent和Skill"""
    global agents, _skill_registry

    # 初始化Skill注册表
    _skill_registry = get_skill_registry("skills")

    # 创建子Agent
    sub_agents = {
        "preference_agent": PreferenceAgent(name="PreferenceAgent"),
        "info_query_agent": InfoQueryAgent(name="InfoQueryAgent"),
        "execution_agent": ExecutionAgent(name="ExecutionAgent"),
        "planning_agent": PlanningAgent(name="PlanningAgent"),
        "summarization_agent": SummarizationAgent(name="SummarizationAgent"),
    }

    # 加载ToolSkill
    tool_skill = load_skill("tool_skill")
    if tool_skill:
        sub_agents["tool_skill"] = tool_skill

    agents = {
        "intention_agent": IntentionAgent(name="IntentionAgent"),
        "orchestration_agent": OrchestrationAgent(
            name="OrchestrationAgent",
            agents=sub_agents
        ),
        "memory_agent": MemoryAgent(name="MemoryAgent"),
    }

    print("✓ All Agents initialized")


# ============================================================
# API端点
# ============================================================

@router.post("/chat")
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """
    处理用户聊天请求
    完整流程: 意图识别 → 调度 → 执行 → 记忆更新

    特性:
    - 请求频率限制
    - 消息长度验证
    - 超时保护
    - 对话上下文传递
    - 详细错误处理
    """
    global agents

    # 获取客户端标识
    client_ip = http_request.client.host if http_request.client else "unknown"
    user_id = request.user_id or "default"
    rate_key = f"{client_ip}:{user_id}"

    # 1. 频率检查
    allowed, remaining_min, remaining_day, limit_msg = _rate_limiter.is_allowed(rate_key)
    if not allowed:
        return ChatResponse(
            response=json.dumps({"error": limit_msg, "retry_after": "1 minute"}),
            session_id=request.session_id or str(uuid.uuid4()),
            tasks_executed=[],
            memory_updated=False
        )

    # 2. 初始化Agent（延迟初始化）
    if not agents:
        init_agents()

    session_id = request.session_id or str(uuid.uuid4())

    # 3. 消息验证
    is_valid, error_msg = validate_message(request.message, max_length=2000)
    if not is_valid:
        return ChatResponse(
            response=json.dumps({"error": f"消息验证失败: {error_msg}"}),
            session_id=session_id,
            tasks_executed=[],
            memory_updated=False
        )

    start_time = time.time()

    try:
        # ========================================
        # Step 0: 获取对话上下文
        # ========================================
        conversation_context = await _history_manager.get_context(session_id, max_turns=5)
        turn_count = await _history_manager.get_turn_count(session_id)

        # ========================================
        # Step 1: 意图识别（带上下文）
        # ========================================
        # 构建带上下文的意图识别消息
        if conversation_context:
            intent_prompt = f"""【对话历史】
{conversation_context}

【当前消息】
{request.message}

请根据对话上下文识别当前消息的意图。"""
        else:
            intent_prompt = request.message

        intention_msg = Msg(name="user", content=intent_prompt, role="user")

        try:
            intention_result = await asyncio.wait_for(
                agents["intention_agent"].reply(intention_msg),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            return ChatResponse(
                response=json.dumps({"error": "意图识别超时，请稍后重试"}),
                session_id=session_id,
                tasks_executed=["intention_agent"],
                memory_updated=False
            )

        try:
            intent_data = json.loads(intention_result.content) if isinstance(intention_result.content, str) else {}
        except json.JSONDecodeError:
            intent_data = {"intent": "general_chat", "entities": {}}

        # 添加上下文信息到 entities
        if conversation_context:
            intent_data["has_context"] = True
            intent_data["turn_count"] = turn_count

        # ========================================
        # Step 2: 调度执行
        # ========================================
        try:
            orchestration_result = await asyncio.wait_for(
                agents["orchestration_agent"].reply(intention_result),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            return ChatResponse(
                response=json.dumps({"error": "处理超时，请稍后重试"}),
                session_id=session_id,
                tasks_executed=[intent_data.get("intent", "unknown"), "orchestration"],
                memory_updated=False
            )

        # 解析调度结果
        try:
            orch_data = json.loads(orchestration_result.content) if isinstance(orchestration_result.content, str) else {}
            final_response = orch_data.get("final_response", str(orchestration_result.content))
        except json.JSONDecodeError:
            final_response = str(orchestration_result.content)
            orch_data = {}

        # ========================================
        # Step 3: 保存对话历史
        # ========================================
        await _history_manager.add_message(
            session_id=session_id,
            role="user",
            content=request.message,
            metadata={"intent": intent_data.get("intent")}
        )

        assistant_response = final_response
        await _history_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=assistant_response,
            metadata={"orch_data": orch_data}
        )

        # ========================================
        # Step 4: 记忆更新（异步，不阻塞）
        # ========================================
        memory_intents = ["travel_planning", "preference_manage", "event_collection"]
        memory_updated = False

        if intent_data.get("intent") in memory_intents:
            memory_data = {
                "content": request.message,
                "session_id": session_id,
                "user_id": user_id,
                "intent": intent_data.get("intent", "unknown"),
                "entities": intent_data.get("entities", {}),
                "timestamp": time.time()
            }
            try:
                asyncio.create_task(
                    agents["memory_agent"].reply(Msg(name="user", content=json.dumps(memory_data), role="user"))
                )
                memory_updated = True
            except Exception:
                pass

        elapsed_ms = int((time.time() - start_time) * 1000)

        return ChatResponse(
            response=json.dumps({
                "response": final_response,
                "elapsed_ms": elapsed_ms,
                "intent": intent_data.get("intent", "unknown"),
                "has_context": bool(conversation_context),
                "turn_count": turn_count + 1,
                "remaining_minute": remaining_min,
                "remaining_day": remaining_day
            }, ensure_ascii=False),
            session_id=session_id,
            tasks_executed=[intent_data.get("intent", "unknown"), "orchestration"],
            memory_updated=memory_updated
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return ChatResponse(
            response=json.dumps({
                "error": f"服务出现问题: {str(e)[:100]}",
                "elapsed_ms": elapsed_ms
            }),
            session_id=session_id,
            tasks_executed=[],
            memory_updated=False
        )


@router.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "agents_initialized": len(agents) > 0,
        "timestamp": time.time()
    }


@router.get("/sessions/{session_id}/history")
async def get_history(session_id: str, limit: int = 50):
    """获取会话历史"""
    try:
        history = await _history_manager.get_history(session_id, limit)
        turn_count = await _history_manager.get_turn_count(session_id)
        return {
            "history": history,
            "session_id": session_id,
            "turn_count": turn_count
        }
    except Exception as e:
        return {"history": [], "session_id": session_id, "error": str(e)}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    try:
        await _history_manager.clear_history(session_id)
        from memory.short_term import ShortTermMemory
        memory = ShortTermMemory()
        memory.clear_session(session_id)
        return {"success": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/context")
async def get_context(session_id: str, max_turns: int = 5):
    """获取对话上下文（用于调试）"""
    context = await _history_manager.get_context(session_id, max_turns)
    turn_count = await _history_manager.get_turn_count(session_id)
    return {
        "session_id": session_id,
        "context": context,
        "turn_count": turn_count,
        "max_turns": max_turns
    }


@router.get("/skills")
async def list_skills():
    """列出所有可用Skill"""
    global _skill_registry
    if not _skill_registry:
        _skill_registry = get_skill_registry("skills")

    return {
        "skills": [
            {
                "name": s.name,
                "version": s.version,
                "description": s.description,
                "agent_type": s.agent_type,
                "tools": s.tools
            }
            for s in _skill_registry.list_skills().values()
        ]
    }


@router.post("/skill/{skill_name}/invoke")
async def invoke_skill(skill_name: str, params: dict):
    """直接调用指定Skill"""
    skill_agent = load_skill(skill_name)
    if not skill_agent:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

    query = params.get("query", "")
    try:
        result = await asyncio.wait_for(
            skill_agent.reply(Msg(name="user", content=query, role="user")),
            timeout=30.0
        )
        return json.loads(result.content) if isinstance(result.content, str) else result.content
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Skill执行超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rate-limit-status")
async def get_rate_limit_status(http_request: Request, user_id: str = "default"):
    """获取当前限流状态"""
    client_ip = http_request.client.host if http_request.client else "unknown"
    rate_key = f"{client_ip}:{user_id}"

    _, remaining_min, remaining_day, _ = _rate_limiter.is_allowed(rate_key, max_per_minute=60, max_per_day=1000)

    return {
        "ip": client_ip,
        "user_id": user_id,
        "remaining_minute": remaining_min,
        "remaining_day": remaining_day,
        "limit_per_minute": 60,
        "limit_per_day": 1000
    }