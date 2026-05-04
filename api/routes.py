"""
FastAPI路由定义
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from models.schemas import ChatRequest, ChatResponse, IntentType
from typing import Dict, Optional
import uuid
import json
import time

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

# 创建路由
router = APIRouter(prefix="/api")

# 全局Agent实例
agents: Dict = {}
_skill_registry = None


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


@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    处理用户聊天请求
    完整流程: 意图识别 → 调度 → 执行 → 记忆更新
    """
    if not agents:
        init_agents()

    session_id = request.session_id or str(uuid.uuid4())

    try:
        # Step 1: 意图识别
        intention_msg = Msg(name="user", content=request.message, role="user")
        intention_result = await agents["intention_agent"].reply(intention_msg)
        intent_data = json.loads(intention_result.content) if isinstance(intention_result.content, str) else {}

        # Step 2: 调度执行
        orchestration_result = await agents["orchestration_agent"].reply(intention_result)

        # Step 3: 记忆更新 (只在特定意图下调用)
        memory_intents = ["travel_planning", "preference_manage", "event_collection"]
        if intent_data.get("intent") in memory_intents:
            memory_data = {
                "content": request.message,
                "session_id": session_id,
                "user_id": request.user_id or "default",
                "intent": intent_data.get("intent", "unknown"),
                "entities": intent_data.get("entities", {}),
                "timestamp": time.time()
            }
            await agents["memory_agent"].reply(Msg(name="user", content=json.dumps(memory_data), role="user"))

        # 解析结果
        orch_data = json.loads(orchestration_result.content) if isinstance(orchestration_result.content, str) else {}
        final_response = orch_data.get("final_response", str(orchestration_result.content))

        return ChatResponse(
            response=json.dumps(orch_data, ensure_ascii=False),
            session_id=session_id,
            tasks_executed=[intent_data.get("intent", "unknown"), "orchestration"],
            memory_updated=True
        )

    except Exception as e:
        return ChatResponse(
            response=json.dumps({"error": str(e)}),
            session_id=session_id,
            tasks_executed=[],
            memory_updated=False
        )


@router.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "agents_initialized": len(agents) > 0
    }


@router.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    """获取会话历史"""
    from memory.short_term import ShortTermMemory
    memory = ShortTermMemory()
    return {"history": memory.get(session_id)}


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
    result = await skill_agent.reply(Msg(name="user", content=query, role="user"))
    return json.loads(result.content) if isinstance(result.content, str) else result.content