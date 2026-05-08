"""
日程规划Agent - 生成完整出行规划
根据用户需求生成合理的行程安排
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
from typing import Optional, Union, List, Dict

from core.llm_client import llm_chat
from core.utils import safe_json_parse


class PlanningAgent(AgentBase):
    """
    日程规划Agent - 生成完整出行规划
    职责：
    1. 根据目的地和时间生成行程安排
    2. 整合偏好信息（酒店、交通、餐饮）
    3. 提供时间规划和注意事项
    4. 生成预算估算
    """

    SYSTEM_PROMPT = """你是一个专业的旅行规划助手，负责生成完整合理的行程安排。

## 输入信息

你会收到以下信息（通过 p1_results 传入）：
1. **用户需求** (query): 用户描述的旅行需求
2. **实体信息** (entities): 识别的地点、时间、人数等
3. **P1结果** (p1_results):
   - info_query_agent: 天气、交通等实时信息
   - preference_agent: 用户偏好（酒店品牌、交通方式、餐饮等）
   - execution_agent: 已执行的订票/预约信息

## 输出格式

生成JSON格式的行程规划：
{
    "itinerary": {
        "day_1": {
            "date": "日期",
            "theme": "主题（如：商务/休闲/探索）",
            "activities": [
                {
                    "time": "09:00",
                    "activity": "活动名称",
                    "location": "地点",
                    "duration": "2小时",
                    "tips": "注意事项"
                }
            ],
            "meals": {
                "breakfast": {"place": "餐厅", "recommendation": "推荐菜"},
                "lunch": {"place": "餐厅", "recommendation": "推荐菜"},
                "dinner": {"place": "餐厅", "recommendation": "推荐菜"}
            },
            "transport": {"from": "出发点", "to": "目的地", "method": "交通方式", "duration": "30分钟"}
        }
    },
    "summary": "行程概览（2-3句话）",
    "budget_estimate": {
        "total": "总预算",
        "breakdown": {"交通": "XX", "住宿": "XX", "餐饮": "XX", "门票": "XX"}
    },
    "tips": ["注意事项1", "注意事项2", "注意事项3"],
    "packing_list": ["必要物品1", "必要物品2"]
}

## 整合偏好

根据 preference_agent 提供的偏好进行推荐：
- **酒店偏好**：优先推荐用户喜欢的酒店品牌（如汉庭、如家、万豪等）
- **交通偏好**：根据用户偏好安排出行方式（地铁/打车/公交）
- **餐饮偏好**：根据用户喜好推荐餐厅类型（中餐/火锅/海鲜等）

## 整合实时信息

根据 info_query_agent 提供的信息：
- **天气**：根据天气情况安排活动（如雨天安排室内活动）
- **交通**：根据路况调整出行时间和路线

## 生成原则

1. **合理性**：行程安排符合逻辑，避免过于紧凑
2. **个性化**：充分利用用户偏好信息
3. **实用性**：提供有用的 tips 和 packing_list
4. **完整性**：覆盖吃住行游购娱各方面"""

    def __init__(self, name: str = "PlanningAgent", model_config: dict = None, **kwargs):
        super().__init__()
        self.name = name
        self.model_config = model_config or {}
        self.memory = InMemoryMemory()

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """处理日程规划请求"""
        if x is None:
            return Msg(name=self.name, content=json.dumps({"error": "No input"}), role="assistant")

        if isinstance(x, list):
            x = x[-1]

        try:
            data = json.loads(x.content) if isinstance(x.content, str) else x.content
        except:
            data = {"query": str(x.content)}

        # 生成行程规划
        plan = await self._generate_plan(data)

        return Msg(
            name=self.name,
            content=json.dumps(plan, ensure_ascii=False),
            role="assistant"
        )

    async def _generate_plan(self, data: dict) -> Dict:
        """使用LLM生成行程规划"""
        query = data.get("query", "")
        entities = data.get("entities", {})
        p1_results = data.get("p1_results", {})  # Priority 1的Agent结果

        # 收集上下文信息
        context_parts = [f"用户需求: {query}"]

        # 添加目的地
        if entities.get("locations"):
            context_parts.append(f"目的地: {', '.join(entities['locations'])}")

        # 添加时间
        if entities.get("date"):
            context_parts.append(f"出行时间: {entities['date']}")

        # 添加偏好信息
        if p1_results:
            # 从InfoQueryAgent获取天气信息
            if "info_query_agent" in p1_results:
                weather_info = p1_results["info_query_agent"].get("response", "")
                if weather_info:
                    context_parts.append(f"天气信息: {weather_info}")

            # 从MemoryAgent获取匹配到的用户偏好（新版架构）
            if "memory_agent" in p1_results:
                memory_result = p1_results["memory_agent"]
                if isinstance(memory_result, dict):
                    matched_prefs = memory_result.get("matched_preferences", [])
                    if matched_prefs:
                        pref_str = self._format_matched_preferences(matched_prefs)
                        context_parts.append(f"用户偏好:\n{pref_str}")

        # 构建提示
        context = "\n".join(context_parts)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"请为以下需求生成详细行程规划:\n{context}"}
        ]

        try:
            response = await llm_chat(messages)
            plan = safe_json_parse(response)

            if plan is None:
                return {
                    "action": "planning",
                    "error": "JSON解析失败",
                    "response": "抱歉，生成行程时出现问题"
                }

            # 构建响应
            return {
                "action": "planning",
                "itinerary": plan.get("itinerary", {}),
                "summary": plan.get("summary", ""),
                "budget_estimate": plan.get("budget_estimate", {}),
                "tips": plan.get("tips", []),
                "packing_list": plan.get("packing_list", []),
                "response": self._format_plan_response(plan)
            }

        except Exception as e:
            return {
                "action": "planning",
                "error": str(e),
                "response": "生成行程时出现问题"
            }

    def _format_matched_preferences(self, matched_prefs: List[Dict]) -> str:
        """格式化从MemoryAgent匹配到的偏好"""
        if not matched_prefs:
            return ""

        formatted_parts = []
        for pref in matched_prefs:
            # 从metadata中提取description或value
            value = pref.get("preference_value", "")
            metadata = pref.get("metadata", {}) or {}

            # 优先级：metadata.description > value > key
            if metadata.get("description"):
                desc = metadata["description"]
            elif value:
                desc = f"{pref.get('category', '')}:{value}"
            else:
                desc = f"{pref.get('category', '')}:{pref.get('preference_key', '')}"

            # 添加置信度
            conf = pref.get("confidence", 0.5)
            formatted_parts.append(f"- {desc} (置信度: {conf:.0%})")

        return "\n".join(formatted_parts) if formatted_parts else ""

    def _format_plan_response(self, plan: Dict) -> str:
        """格式化行程响应为友好文本"""
        itinerary = plan.get("itinerary", {})
        summary = plan.get("summary", "")

        if not itinerary:
            return "抱歉，无法生成行程规划"

        # 简单格式化
        days = list(itinerary.keys())
        if len(days) == 1:
            return f"已为您生成1天行程：{summary}"
        elif len(days) > 1:
            return f"已为您生成{len(days)}天行程：{summary}"

        return summary

    # ==================== Skill预留位置 ====================

    # TODO: budget_planning_skill - 预算规划Skill
    # TODO: route_optimization_skill - 路线优化Skill
    # TODO: packing_suggestion_skill - 行李建议Skill
    # TODO: local_guide_skill - 当地向导Skill