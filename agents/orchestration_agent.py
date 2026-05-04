"""
调度Agent - 优先级+并行混合调度模式
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
import asyncio
from typing import Optional, Union, List, Dict, Any

from core.llm_client import llm_chat


class OrchestrationAgent(AgentBase):
    """
    调度Agent - 核心调度器
    实现优先级+并行混合调度模式
    - Priority 1: 可并行执行（偏好查询、信息查询、外部执行）
    - Priority 2: 依赖P1结果（日程规划、对话总结）
    """

    def __init__(
        self,
        name: str = "OrchestrationAgent",
        model_config: dict = None,
        agents: dict = None,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.model_config = model_config or {}
        self.memory = InMemoryMemory()
        self.agents = agents or {}

        # 意图到Agent的映射
        self.intent_agent_map = {
            "travel_planning": {
                "p1": ["info_query_agent"],
                "p2": ["planning_agent", "summarization_agent"]
            },
            "info_query": {
                "p1": ["info_query_agent"],
                "p2": []
            },
            "preference_manage": {
                "p1": ["preference_agent"],
                "p2": []
            },
            "memory_query": {
                "p1": ["preference_agent"],
                "p2": ["summarization_agent"]
            },
            "event_collection": {
                "p1": ["preference_agent", "info_query_agent"],
                "p2": ["planning_agent"]
            },
            "execution": {
                "p1": ["execution_agent"],
                "p2": []
            },
            "general_chat": {
                "p1": [],
                "p2": ["summarization_agent"]
            }
        }

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """执行调度逻辑"""
        if x is None:
            return Msg(name=self.name, content=json.dumps({"error": "No input"}), role="assistant")

        if isinstance(x, list):
            x = x[-1]

        # 解析意图结果
        try:
            intent_data = json.loads(x.content) if isinstance(x.content, str) else x.content
        except:
            intent_data = {"intent": "unknown", "entities": {}}

        intent = intent_data.get("intent", "unknown")
        entities = intent_data.get("entities", {})
        query = intent_data.get("query", "")

        # 执行调度
        results = await self._dispatch_tasks(intent, entities, query)

        return Msg(
            name=self.name,
            content=json.dumps(results, ensure_ascii=False),
            role="assistant"
        )

    async def _dispatch_tasks(
        self,
        intent: str,
        entities: dict,
        query: str
    ) -> dict:
        """分发任务到各个Agent"""

        all_results = {
            "intent": intent,
            "entities": entities,
            "query": query,
            "priority_1_results": {},
            "priority_2_results": {},
            "final_response": ""
        }

        # 获取任务映射
        task_map = self.intent_agent_map.get(intent, {"p1": [], "p2": []})
        p1_agents = task_map.get("p1", [])
        p2_agents = task_map.get("p2", [])

        # Priority 1: 并行执行
        if p1_agents:
            p1_tasks = [(agent, query) for agent in p1_agents]
            p1_results = await self._execute_parallel(p1_tasks)
            all_results["priority_1_results"] = p1_results

        # Priority 2: 依赖P1结果
        if p2_agents:
            p2_input = {
                "query": query,
                "entities": entities,
                "p1_results": all_results["priority_1_results"]
            }
            p2_tasks = [(agent, p2_input) for agent in p2_agents]
            p2_results = await self._execute_parallel(p2_tasks)
            all_results["priority_2_results"] = p2_results

        # 生成最终响应
        all_results["final_response"] = self._generate_response(all_results)

        return all_results

    async def _execute_parallel(self, tasks: list) -> dict:
        """并行执行任务列表"""
        async def execute_task(agent_name: str, task_input: Any):
            agent = self.agents.get(agent_name)
            if agent is None:
                return {"error": f"Agent {agent_name} not found"}

            try:
                if isinstance(task_input, dict):
                    content = json.dumps(task_input, ensure_ascii=False)
                else:
                    content = str(task_input)

                result = await agent.reply(Msg(
                    name="Orchestrator",
                    content=content,
                    role="user"
                ))

                parsed = json.loads(result.content) if isinstance(result.content, str) else result.content
                return parsed

            except Exception as e:
                return {"error": f"{agent_name}: {str(e)}"}

        # 使用asyncio.gather并行执行
        results = await asyncio.gather(
            *[execute_task(agent_name, task_input) for agent_name, task_input in tasks],
            return_exceptions=True
        )

        return {
            tasks[i][0]: results[i]
            for i in range(len(tasks))
        }

    def _generate_response(self, results: dict) -> str:
        """生成最终响应"""
        p1 = results.get("priority_1_results", {})
        p2 = results.get("priority_2_results", {})
        intent = results.get("intent", "unknown")

        all_results = {**p1, **p2}

        # 收集有效响应
        response_parts = []
        for agent_name, result in all_results.items():
            if isinstance(result, dict) and "response" in result:
                response_parts.append(result["response"])
            elif isinstance(result, dict) and "itinerary" in result:
                response_parts.append("行程规划已完成")

        if response_parts:
            return " | ".join(response_parts)

        # 默认响应
        default_responses = {
            "travel_planning": "行程规划已完成",
            "info_query": "信息查询完成",
            "preference_manage": "偏好已更新",
            "memory_query": "记忆查询完成",
            "event_collection": "已收集行程要素",
            "execution": "任务执行完成",
            "general_chat": "你好！有什么可以帮助您的吗？"
        }

        return default_responses.get(intent, "已完成处理")