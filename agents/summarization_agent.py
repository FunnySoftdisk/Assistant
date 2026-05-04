"""
总结对话Agent - 使用LLM总结对话内容
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
from typing import Optional, Union, List, Dict

from core.llm_client import llm_chat
from core.utils import safe_json_parse


class SummarizationAgent(AgentBase):
    """
    总结对话Agent - 总结对话要点，提取关键信息
    """

    SYSTEM_PROMPT = """你是一个对话总结助手，负责提炼对话的关键信息。

请分析对话内容，提取：
1. 用户的主要意图
2. 已收集的关键信息（地点、时间、偏好等）
3. 已执行的操作
4. 后续待办事项

输出JSON格式：
{
    "main_intent": "主要意图",
    "entities_collected": ["实体1", "实体2"],
    "actions_taken": ["操作1", "操作2"],
    "pending_tasks": ["待办1", "待办2"],
    "summary": "简洁总结"
}"""

    def __init__(self, name: str = "SummarizationAgent", model_config: dict = None, **kwargs):
        super().__init__()
        self.name = name
        self.model_config = model_config or {}
        self.memory = InMemoryMemory()

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

        # 生成总结
        summary = await self._summarize(data)

        return Msg(
            name=self.name,
            content=json.dumps(summary, ensure_ascii=False),
            role="assistant"
        )

    async def _summarize(self, data: dict) -> Dict:
        """使用LLM生成总结"""
        query = data.get("query", "")
        p1_results = data.get("p1_results", {})
        p2_results = data.get("p2_results", {})

        # 构建上下文
        context_parts = [f"用户请求: {query}"]

        # 添加各Agent结果
        all_results = {**p1_results, **p2_results}
        for agent, result in all_results.items():
            if isinstance(result, dict) and "response" in result:
                context_parts.append(f"- {agent}: {result['response']}")

        context = "\n".join(context_parts)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"请总结以下对话:\n{context}"}
        ]

        try:
            response = await llm_chat(messages)

            # 安全解析JSON
            summary = safe_json_parse(response)

            if summary is None:
                return {
                    "action": "summarize",
                    "error": "JSON解析失败",
                    "response": f"你好！{query}，有什么可以帮助您的吗？"
                }

            return {
                "action": "summarize",
                "main_intent": summary.get("main_intent", query),
                "entities_collected": summary.get("entities_collected", []),
                "actions_taken": summary.get("actions_taken", list(all_results.keys())),
                "pending_tasks": summary.get("pending_tasks", []),
                "summary": summary.get("summary", ""),
                "response": summary.get("summary", f"你好！有什么可以帮助您的？")
            }

        except Exception as e:
            return {
                "action": "summarize",
                "error": str(e),
                "response": f"[LLM调用失败] {str(e)}"
            }