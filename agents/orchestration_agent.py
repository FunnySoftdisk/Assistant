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
from skills.generic_skill import get_generic_skill_loader


class OrchestrationAgent(AgentBase):
    """
    调度Agent - 核心调度器
    实现优先级+并行混合调度模式

    调度规则：
    - Priority 1 (P1): 可并行执行
      - memory_agent: 匹配用户偏好
      - info_query_agent: 查询天气/交通等外部信息
      - preference_agent: 查询/更新偏好

    - Priority 2 (P2): 依赖P1结果
      - planning_agent: 生成行程规划（需要等P1拿到偏好）
      - summarization_agent: 总结对话

    - Priority 3 (P3): 依赖P2结果
      - execution_agent: 执行操作（需要等Planning生成具体计划后才能执行）

    特性：
    - 单个Agent失败不影响其他Agent
    - 超时控制（单个Agent最多30秒）
    - 结果聚合优化
    - 优雅降级
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

        # 加载通用Skills
        self.skill_loader = get_generic_skill_loader()
        print(f"✓ 已加载 {len(self.skill_loader.list_skills())} 个Skills")

        # 意图到Agent的映射
        # P1: 并行获取信息和偏好
        # P2: 依赖P1结果生成规划
        # P3: 依赖P2结果执行操作
        # Summarization: 整合所有结果告诉用户
        self.intent_agent_map = {
            "travel_planning": {
                "p1": ["memory_agent", "info_query_agent"],
                "p2": ["planning_agent"],
                "p3": ["execution_agent"],
                "summarize": ["summarization_agent"]
            },
            "info_query": {
                "p1": ["info_query_agent"],
                "p2": [],
                "p3": [],
                "summarize": ["summarization_agent"]
            },
            "preference_manage": {
                "p1": ["memory_agent", "preference_agent"],
                "p2": [],
                "p3": [],
                "summarize": ["summarization_agent"]
            },
            "memory_query": {
                "p1": ["memory_agent"],
                "p2": [],
                "p3": [],
                "summarize": ["summarization_agent"]
            },
            "event_collection": {
                "p1": ["memory_agent", "info_query_agent"],
                "p2": ["planning_agent"],
                "p3": ["execution_agent"],
                "summarize": ["summarization_agent"]
            },
            "execution": {
                "p1": ["memory_agent"],
                "p2": [],
                "p3": ["execution_agent"],
                "summarize": ["summarization_agent"]
            },
            "general_chat": {
                "p1": [],
                "p2": [],
                "p3": [],
                "summarize": ["summarization_agent"]
            }
        }

        # Agent超时配置
        self.agent_timeout = 30.0

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
        try:
            results = await self._dispatch_tasks(intent, entities, query)
        except asyncio.TimeoutError:
            results = {
                "intent": intent,
                "entities": entities,
                "query": query,
                "error": "调度超时，部分结果可能不完整",
                "priority_1_results": intent_data.get("p1_results", {}),
                "priority_2_results": {},
                "priority_3_results": {},
                "final_response": "服务处理中部分超时，请稍后重试"
            }
        except Exception as e:
            results = {
                "intent": intent,
                "entities": entities,
                "query": query,
                "error": str(e),
                "priority_1_results": {},
                "priority_2_results": {},
                "priority_3_results": {},
                "final_response": f"处理出现问题: {str(e)[:100]}"
            }

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
        """分发任务到各个Agent，支持P1/P2/P3+Summarize多级调度"""

        all_results = {
            "intent": intent,
            "entities": entities,
            "query": query,
            "priority_1_results": {},
            "priority_2_results": {},
            "priority_3_results": {},
            "summarize_results": {},
            "final_response": ""
        }

        # 获取任务映射
        task_map = self.intent_agent_map.get(intent, {"p1": [], "p2": [], "p3": [], "summarize": []})
        p1_agents = task_map.get("p1", [])
        p2_agents = task_map.get("p2", [])
        p3_agents = task_map.get("p3", [])
        summarize_agents = task_map.get("summarize", [])

        # ============================================
        # Priority 1: 并行执行
        # ============================================
        if p1_agents:
            p1_tasks = self._build_p1_tasks(p1_agents, query, entities)
            p1_results = await self._execute_parallel(p1_tasks, timeout=self.agent_timeout)
            all_results["priority_1_results"] = p1_results

            # 从P1结果中提取匹配到的偏好（供后续使用）
            matched_prefs = self._extract_matched_preferences(p1_results)
            entities["_matched_preferences"] = matched_prefs

        # ============================================
        # Priority 2: 依赖P1结果
        # ============================================
        if p2_agents:
            p2_input = {
                "query": query,
                "entities": entities,
                "p1_results": all_results["priority_1_results"]
            }
            p2_tasks = [(agent, p2_input) for agent in p2_agents]
            p2_results = await self._execute_parallel(p2_tasks, timeout=self.agent_timeout)
            all_results["priority_2_results"] = p2_results

            # 从P2结果中提取行程规划（供执行使用）
            planning_output = self._extract_planning_output(p2_results)
            entities["_planning_output"] = planning_output

        # ============================================
        # Priority 3: 依赖P2结果（用于执行操作）
        # ============================================
        if p3_agents:
            p3_input = {
                "query": query,
                "entities": entities,
                "p1_results": all_results["priority_1_results"],
                "p2_results": all_results["priority_2_results"]
            }
            p3_tasks = [(agent, p3_input) for agent in p3_agents]
            p3_results = await self._execute_parallel(p3_tasks, timeout=self.agent_timeout)
            all_results["priority_3_results"] = p3_results

        # ============================================
        # Summarize: 整合所有结果，告诉用户完整情况
        # 在P3执行完成后，调用SummarizationAgent整合信息
        # ============================================
        if summarize_agents:
            summarize_input = {
                "query": query,
                "entities": entities,
                "p1_results": all_results["priority_1_results"],
                "p2_results": all_results["priority_2_results"],
                "p3_results": all_results["priority_3_results"]
            }
            summarize_tasks = [(agent, summarize_input) for agent in summarize_agents]
            summarize_results = await self._execute_parallel(summarize_tasks, timeout=self.agent_timeout)
            all_results["summarize_results"] = summarize_results

        # 生成最终响应
        all_results["final_response"] = self._generate_response(all_results)

        return all_results

    def _build_p1_tasks(self, p1_agents: List[str], query: str, entities: dict) -> List[tuple]:
        """构建P1任务列表"""
        tasks = []
        for agent in p1_agents:
            if agent == "memory_agent":
                tasks.append((agent, {
                    "action": "match",
                    "query": query,
                    "entities": entities,
                    "user_id": "default",
                    "session_id": entities.get("session_id", "default")
                }))
            elif agent == "info_query_agent":
                tasks.append((agent, query))
            elif agent == "preference_agent":
                tasks.append((agent, query))
            else:
                tasks.append((agent, query))
        return tasks

    def _extract_matched_preferences(self, p1_results: dict) -> Dict:
        """从P1结果中提取匹配到的偏好"""
        if "memory_agent" in p1_results:
            memory_result = p1_results["memory_agent"]
            if isinstance(memory_result, dict):
                return memory_result.get("matched_preferences", [])
        return []

    def _extract_planning_output(self, p2_results: dict) -> Dict:
        """从P2结果中提取行程规划输出"""
        if "planning_agent" in p2_results:
            planning_result = p2_results["planning_agent"]
            if isinstance(planning_result, dict):
                return {
                    "itinerary": planning_result.get("itinerary", {}),
                    "response": planning_result.get("response", "")
                }
        return {}

    async def _execute_parallel(self, tasks: list, timeout: float = 30.0) -> dict:
        """并行执行任务列表，单个失败不影响其他"""

        async def execute_task_safe(agent_name: str, task_input: Any) -> tuple:
            """安全的任务执行包装器"""
            agent = self.agents.get(agent_name)
            if agent is None:
                return agent_name, {"error": f"Agent {agent_name} not found"}

            try:
                if isinstance(task_input, dict):
                    content = json.dumps(task_input, ensure_ascii=False)
                else:
                    content = str(task_input)

                result = await asyncio.wait_for(
                    agent.reply(Msg(name="Orchestrator", content=content, role="user")),
                    timeout=timeout
                )

                parsed = json.loads(result.content) if isinstance(result.content, str) else result.content
                return agent_name, parsed

            except asyncio.TimeoutError:
                return agent_name, {"error": f"{agent_name} 执行超时", "timeout": True}
            except json.JSONDecodeError as e:
                return agent_name, {"error": f"{agent_name} JSON解析失败: {str(e)}"}
            except Exception as e:
                return agent_name, {"error": f"{agent_name} 执行失败: {str(e)}"}

        results = await asyncio.gather(
            *[execute_task_safe(agent_name, task_input) for agent_name, task_input in tasks],
            return_exceptions=True
        )

        output = {}
        for i, result in enumerate(results):
            agent_name = tasks[i][0]
            if isinstance(result, Exception):
                output[agent_name] = {"error": f"任务执行异常: {str(result)}"}
            else:
                agent_name_result, parsed = result
                output[agent_name] = parsed

        return output

    def _generate_response(self, results: dict) -> str:
        """生成最终响应"""
        p1 = results.get("priority_1_results", {})
        p2 = results.get("priority_2_results", {})
        p3 = results.get("priority_3_results", {})
        summarize = results.get("summarize_results", {})
        intent = results.get("intent", "unknown")

        # 优先使用SummarizationAgent的总结结果（告诉用户完整情况）
        if summarize:
            for agent_name, result in summarize.items():
                if isinstance(result, dict) and result.get("response"):
                    return result["response"]

        # 如果没有summarization，回退到各Agent结果的组合
        all_results = {**p1, **p2, **p3}

        response_parts = []
        for agent_name, result in all_results.items():
            if isinstance(result, dict):
                if "response" in result and result["response"]:
                    response_parts.append(result["response"])
                elif "itinerary" in result and agent_name == "planning_agent":
                    response_parts.append("行程规划已完成")
                elif "status" in result and agent_name == "execution_agent":
                    status = result.get("status")
                    if status == "success":
                        response_parts.append(f"执行成功: {result.get('response', '')}")
                    elif status == "simulated":
                        response_parts.append(f"模拟操作: {result.get('response', '')}")
                    else:
                        response_parts.append(f"执行状态: {status}")

        if response_parts:
            return " | ".join(response_parts)

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
