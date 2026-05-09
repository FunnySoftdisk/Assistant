"""
总结对话Agent - 整合所有结果告诉用户完整情况
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
import asyncio
import re
from typing import Optional, Union, List, Dict

from core.llm_client import llm_chat
from core.utils import safe_json_parse


class SummarizationAgent(AgentBase):
    """
    总结对话Agent - 整合所有Agent结果，告诉用户完整情况

    在P3执行完成后调用，整合：
    - P1: 用户偏好匹配结果、天气查询结果
    - P2: 行程规划结果
    - P3: 外部执行结果（订票、闹钟等）

    特性:
    - LLM调用超时保护（30秒）
    - JSON解析失败时返回降级响应
    - 重点关注执行结果的反馈
    - 动态感知系统Skill能力
    """

    def __init__(self, name: str = "SummarizationAgent", model_config: dict = None, **kwargs):
        super().__init__()
        self.name = name
        self.model_config = model_config or {}
        self.memory = InMemoryMemory()

        # 加载Skill加载器获取真实能力列表
        from skills.generic_skill import get_generic_skill_loader
        self.skill_loader = get_generic_skill_loader()

        # 动态更新SYSTEM_PROMPT，加入真实Skills
        self._update_skill_prompt()

    def _update_skill_prompt(self):
        """动态更新Skill提示"""
        all_skills = self.skill_loader.list_skills()
        total = len(all_skills)
        with_scripts = sum(1 for s in all_skills.values() if any(t.script_path for t in s.tools))

        skill_lines = []
        for name, skill in all_skills.items():
            if skill.tools and any(t.script_path for t in skill.tools):
                tool_names = [t.name for t in skill.tools if t.script_path]
                if tool_names:
                    skill_lines.append(f"- **{name}**: {', '.join(tool_names)}")

        skills_info = "\n".join(skill_lines) if skill_lines else "无"

        prompt_parts = [
            "你是一个对话总结助手，负责整合对话结果并告诉用户完整情况。",
            "",
            "## 你的职责",
            "1. 整合所有信息：将P1/P2/P3各阶段的结果整合成完整回复",
            "2. 重点反馈执行结果：用户最关心的是外部操作是否成功",
            "3. 清晰展示行程：告诉用户生成的行程安排",
            "4. 提醒后续操作：如需订票确认、出发提醒等",
            "5. 介绍系统能力：如用户问及系统能力，如实介绍已集成的Skills",
            "",
            "## 系统Skill能力",
            f"系统目前集成了 **{total}** 个Skills，其中 **{with_scripts}** 个有可执行工具：",
            skills_info,
            "",
            "## 需要整合的信息",
            "### P1 结果（信息收集）",
            "- 用户偏好匹配结果",
            "- 天气、交通等信息",
            "",
            "### P2 结果（规划生成）",
            "- 行程规划详情",
            "- 预算估算",
            "- 注意事项",
            "",
            "### P3 结果（执行操作）",
            "- 订票结果（成功/失败/模拟）",
            "- 闹钟设置结果",
            "- 通知发送结果",
            "",
            "## 反馈原则",
            "1. 执行结果放首位：先告诉用户订票/闹钟是否成功",
            "2. 行程规划要清晰：用简洁语言描述日程安排",
            "3. 提醒要实用：如出发时间、准备物品等",
            "4. 诚实告知模拟：如果是模拟操作，明确告知用户这是测试",
            "5. 如实介绍能力：当用户问及系统能力时，介绍已集成的真实Skills",
        ]

        self.SYSTEM_PROMPT = "\n".join(prompt_parts)

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """处理总结请求"""
        if x is None:
            return Msg(name=self.name, content=json.dumps({"error": "No input"}), role="assistant")

        if isinstance(x, list):
            x = x[-1]

        try:
            data = json.loads(x.content) if isinstance(x.content, str) else x.content
        except:
            data = {"query": str(x.content)}

        try:
            summary = await self._summarize(data)
        except asyncio.TimeoutError:
            summary = {
                "action": "summarize",
                "error": "LLM超时",
                "response": "总结生成超时，请稍后重试"
            }
        except Exception as e:
            summary = {
                "action": "summarize",
                "error": str(e),
                "response": f"生成总结时出现问题: {str(e)}"
            }

        return Msg(
            name=self.name,
            content=json.dumps(summary, ensure_ascii=False),
            role="assistant"
        )

    async def _summarize(self, data: dict) -> Dict:
        """使用LLM生成总结，整合P1/P2/P3所有结果"""
        query = data.get("query", "")
        p1_results = data.get("p1_results", {})
        p2_results = data.get("p2_results", {})
        p3_results = data.get("p3_results", {})

        # 构建完整上下文
        context_parts = [f"用户请求: {query}"]

        # P1 结果（信息收集）
        if p1_results:
            context_parts.append("\n[P1 信息收集结果]")
            for agent, result in p1_results.items():
                if isinstance(result, dict):
                    response = result.get("response", "")
                    matched = result.get("matched_preferences", [])
                    if response:
                        context_parts.append(f"- {agent}: {response}")
                    if matched:
                        pref_count = len(matched) if isinstance(matched, list) else 0
                        context_parts.append(f"- {agent}: 匹配到 {pref_count} 条相关偏好")

        # P2 结果（行程规划）
        if p2_results:
            context_parts.append("\n[P2 行程规划结果]")
            for agent, result in p2_results.items():
                if isinstance(result, dict):
                    if agent == "planning_agent":
                        itinerary = result.get("itinerary", {})
                        summary = result.get("summary", "")
                        response = result.get("response", "")
                        if itinerary:
                            context_parts.append(f"- 行程规划: {response}")
                            days = list(itinerary.keys())
                            context_parts.append(f"  共 {len(days)} 天行程")
                        elif summary:
                            context_parts.append(f"- 行程概要: {summary}")
                    else:
                        response = result.get("response", "")
                        if response:
                            context_parts.append(f"- {agent}: {response}")

        # P3 结果（执行操作）重点反馈
        if p3_results:
            context_parts.append("\n[P3 执行操作结果]")
            execution_results = []
            for agent, result in p3_results.items():
                if isinstance(result, dict):
                    action = result.get("action", "unknown")
                    status = result.get("status", "unknown")
                    response = result.get("response", "")

                    if status == "success":
                        execution_results.append(f"[OK] {action}成功: {response}")
                    elif status == "simulated":
                        execution_results.append(f"[SIM] {action}(模拟): {response}")
                    elif status == "timeout":
                        execution_results.append(f"[TIMEOUT] {action}超时")
                    elif status == "error":
                        execution_results.append(f"[ERROR] {action}失败: {result.get('error', '')}")

            if execution_results:
                context_parts.extend(execution_results)
            else:
                context_parts.append("- 无执行操作")

        context = "\n".join(context_parts)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"请整合以下对话结果，给用户一个完整的回复:\n{context}"}
        ]

        try:
            response = await asyncio.wait_for(llm_chat(messages), timeout=30.0)

            # 安全解析JSON
            summary = safe_json_parse(response)

            if summary is None:
                # LLM返回的不是JSON格式，可能是自然语言回复
                # 直接使用LLM的响应作为回复（移除LLM思考标签）
                clean_response = response
                if '<think>' in clean_response:
                    clean_response = re.sub(r'<think>.*?', '', clean_response, flags=re.DOTALL)
                return {
                    "action": "summarize",
                    "response": clean_response.strip() if clean_response.strip() else "已完成处理"
                }

            return {
                "action": "summarize",
                "main_intent": summary.get("main_intent", query),
                "planning_summary": summary.get("planning_summary", ""),
                "execution_summary": summary.get("execution_summary", ""),
                "important_reminders": summary.get("important_reminders", []),
                "pending_actions": summary.get("pending_actions", []),
                "next_steps": summary.get("next_steps", ""),
                "response": summary.get("response", self._fallback_response(query, p1_results, p2_results, p3_results))
            }

        except asyncio.TimeoutError:
            return {
                "action": "summarize",
                "error": "LLM超时",
                "response": self._fallback_response(query, p1_results, p2_results, p3_results)
            }
        except Exception as e:
            return {
                "action": "summarize",
                "error": str(e),
                "response": self._fallback_response(query, p1_results, p2_results, p3_results)
            }

    def _fallback_response(self, query: str, p1_results: dict = None, p2_results: dict = None, p3_results: dict = None) -> str:
        """降级响应：当LLM调用失败时的保底响应"""
        parts = []

        # 添加信息查询结果 (p1)
        if p1_results:
            for agent, result in p1_results.items():
                if isinstance(result, dict) and result.get("response"):
                    parts.append(result["response"])

        # 添加行程规划 (p2)
        if p2_results:
            for agent, result in p2_results.items():
                if isinstance(result, dict) and result.get("response"):
                    parts.append(result["response"])

        # 添加执行结果 (p3)
        if p3_results:
            for agent, result in p3_results.items():
                if isinstance(result, dict):
                    status = result.get("status", "")
                    response = result.get("response", "")
                    if status == "success":
                        parts.append(f"执行成功: {response}")
                    elif status == "simulated":
                        parts.append(f"模拟操作: {response}")
                    elif status == "error":
                        parts.append(f"执行失败: {response}")

        if parts:
            return " | ".join(parts)

        return "抱歉，暂时无法处理您的请求，请稍后重试"
