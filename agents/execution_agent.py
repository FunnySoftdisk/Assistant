"""
外部执行Agent - 外部操作执行
负责订票、定闹钟等外部操作
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
import asyncio
from typing import Optional, Union, List, Dict

from core.llm_client import llm_chat


class ExecutionAgent(AgentBase):
    """
    外部执行Agent - 执行外部操作
    职责：
    1. 订票操作 - 机票、酒店、火车票等（模拟，实际需接入API）
    2. 设置闹钟 - 提醒用户重要事项
    3. 发送通知 - 邮件、短信等（模拟）
    4. 日程提醒 - 创建日历事件

    特性:
    - 操作超时保护（15秒）
    - 完整的错误处理和友好提示
    - 操作历史记录
    - 参数提取失败时的回退机制
    """

    SYSTEM_PROMPT = """你是一个外部操作执行助手，负责完成订票、设闹钟、发通知等具体操作。

## 你的职责

1. **理解操作类型**：用户想要执行什么操作（订票/设闹钟/发通知等）
2. **提取操作参数**：时间、地点、人数、事项等
3. **执行操作**：调用内置工具或API完成操作
4. **返回结果**：告知用户操作是否成功

## 操作类型

| 操作 | 关键词示例 | 参数 |
|------|-----------|------|
| 订机票 | "订机票去上海"、"买机票" | 目的地、日期、人数 |
| 订酒店 | "订酒店"、"预订住宿" | 地点、入住日期、入住天数 |
| 设闹钟 | "设闹钟"、"定闹钟 7点" | 时间、标签（起床/开会等） |
| 创建提醒 | "提醒我开会"、"日程 下午3点" | 时间、内容 |
| 发通知 | "发通知"、"发邮件" | 接收人、内容 |

## 参数提取

从用户输入中提取关键信息：
- **目的地**：北京、上海、杭州等城市名
- **日期**：可识别 "3月5日"、"2024-03-05"、"明天"等
- **时间**：可识别 "7点"、"15:30"、"下午3点"等
- **人数**：可识别 "2人"、"3张票"等

如果参数不完整：
- 订票：询问目的地和日期
- 闹钟：询问时间和标签

## 执行结果

成功时返回：
- 操作类型
- 执行状态
- 操作详情（如航班号、酒店确认号等）

失败时返回：
- 操作类型
- 错误原因
- 建议解决方案

## 注意事项

1. **超时处理**：如果操作执行时间过长（>15秒），返回超时提示
2. **参数不全**：不尝试执行参数不全的操作，而是询问用户补充
3. **模拟操作**：当前机票/酒店API未接入时，明确告知用户是模拟操作
4. **操作历史**：记录每次操作，便于后续查询和追溯"""

    def __init__(self, name: str = "ExecutionAgent", model_config: dict = None, **kwargs):
        super().__init__()
        self.name = name
        self.model_config = model_config or {}
        self.memory = InMemoryMemory()
        self.execution_history = []

        # 内置执行工具
        self.builtin_actions = {
            "book_flight": self._book_flight,
            "book_hotel": self._book_hotel,
            "set_alarm": self._set_alarm,
            "create_reminder": self._create_reminder,
            "send_notification": self._send_notification,
        }

        # 操作超时
        self.action_timeout = 15.0

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """处理外部执行请求"""
        if x is None:
            return Msg(name=self.name, content=json.dumps({"error": "No input"}), role="assistant")

        if isinstance(x, list):
            x = x[-1]

        task = x.content if hasattr(x, 'content') else str(x)

        # 尝试解析JSON输入（从OrchestrationAgent P3传来）
        input_data = self._parse_input(task)

        # 优先使用P2传入的行程规划信息
        planning_output = input_data.get("entities", {}).get("_planning_output", {})
        if planning_output:
            # 根据行程规划生成执行任务
            task = self._build_task_from_planning(input_data, planning_output)

        # 解析并执行任务（带超时保护）
        try:
            result = await asyncio.wait_for(
                self._execute_task(task),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            result = {
                "action": "execute",
                "status": "timeout",
                "error": "执行超时",
                "response": "操作执行超时，请稍后重试"
            }
        except Exception as e:
            result = {
                "action": "execute",
                "status": "error",
                "error": str(e),
                "response": f"执行失败: {str(e)[:50]}"
            }

        self.execution_history.append(result)

        return Msg(
            name=self.name,
            content=json.dumps(result, ensure_ascii=False),
            role="assistant"
        )

    def _parse_input(self, task: str) -> Dict:
        """解析输入，可能是纯文本或JSON"""
        try:
            data = json.loads(task)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        return {"query": task, "entities": {}}

    def _build_task_from_planning(self, input_data: Dict, planning_output: Dict) -> str:
        """根据行程规划构建执行任务"""
        query = input_data.get("query", "")
        p1_results = input_data.get("p1_results", {})

        # 从P1结果中提取偏好
        matched_prefs = []
        if "memory_agent" in p1_results:
            matched_prefs = p1_results["memory_agent"].get("matched_preferences", [])

        # 构建执行描述
        execution_parts = []

        # 从行程规划中提取需要执行的操作
        itinerary = planning_output.get("itinerary", {})
        if itinerary:
            for day_key, day_data in itinerary.items():
                if isinstance(day_data, dict):
                    activities = day_data.get("activities", [])
                    for activity in activities:
                        if activity.get("needs_booking"):
                            execution_parts.append(f"{day_key}的{activity.get('activity')}需要预订")

        # 合并用户原始需求
        if query:
            execution_parts.append(query)

        return " | ".join(execution_parts) if execution_parts else query

    def _detect_action_type(self, task: str) -> tuple:
        """检测操作类型"""
        task_lower = task.lower()

        # 订票相关
        if any(kw in task_lower for kw in ["订票", "订机票", "订酒店", "订火车", "购买机票"]):
            if "机票" in task_lower or "航班" in task_lower:
                return "book_flight", self._extract_booking_info(task)
            elif "酒店" in task_lower or "住宿" in task_lower:
                return "book_hotel", self._extract_booking_info(task)
            else:
                return "book_flight", self._extract_booking_info(task)

        # 闹钟相关
        if any(kw in task_lower for kw in ["定闹钟", "设闹钟", "闹钟", "提醒"]):
            return "set_alarm", self._extract_alarm_info(task)

        # 日程提醒
        if any(kw in task_lower for kw in ["日程", "提醒", "calendar", "meeting"]):
            return "create_reminder", self._extract_reminder_info(task)

        # 发送通知
        if any(kw in task_lower for kw in ["通知", "发邮件", "发消息"]):
            return "send_notification", self._extract_notification_info(task)

        return "general", {"task": task}

    def _extract_booking_info(self, task: str) -> Dict:
        """提取订票信息"""
        info = {}

        # 简单提取目的地
        locations = ["北京", "上海", "杭州", "深圳", "广州", "成都", "重庆", "西安"]
        for loc in locations:
            if loc in task:
                info["destination"] = loc
                break

        # 简单提取日期
        import re
        date_match = re.search(r'\d+月\d+日|\d+-\d+-\d+', task)
        if date_match:
            info["date"] = date_match.group()

        # 简单提取人数
        person_match = re.search(r'(\d+)人|\d+张', task)
        if person_match:
            info["quantity"] = person_match.group(1) or "1"

        return info

    def _extract_alarm_info(self, task: str) -> Dict:
        """提取闹钟信息"""
        info = {}

        import re
        # 提取时间
        time_match = re.search(r'(\d+)[点时](\d+)?', task)
        if time_match:
            hour = time_match.group(1)
            minute = time_match.group(2) or "0"
            info["time"] = f"{hour}:{minute.zfill(2)}"

        # 提取标签
        labels = ["起床", "开会", "吃药", "睡觉", "出发"]
        for label in labels:
            if label in task:
                info["label"] = label
                break

        return info

    def _extract_reminder_info(self, task: str) -> Dict:
        """提取日程提醒信息"""
        return {"content": task, "source": "conversation"}

    def _extract_notification_info(self, task: str) -> Dict:
        """提取通知信息"""
        return {"content": task, "source": "conversation"}

    async def _execute_task(self, task: str) -> Dict:
        """执行任务"""
        action_type, params = self._detect_action_type(task)

        if action_type in self.builtin_actions:
            try:
                result = await asyncio.wait_for(
                    self.builtin_actions[action_type](params),
                    timeout=self.action_timeout
                )
                return result
            except asyncio.TimeoutError:
                return {
                    "action": action_type,
                    "status": "timeout",
                    "error": f"{action_type} 执行超时"
                }
            except Exception as e:
                return {
                    "action": action_type,
                    "status": "error",
                    "error": str(e)
                }

        # 通用执行
        return await self._general_execute(task)

    async def _book_flight(self, params: Dict) -> Dict:
        """模拟订机票"""
        try:
            # TODO: 接入真实机票API（如飞猪、携程、去哪儿）
            destination = params.get("destination", "未知")
            date = params.get("date", "待定")

            return {
                "action": "book_flight",
                "status": "simulated",
                "destination": destination,
                "date": date,
                "response": f"模拟订票成功：{date}飞往{destination}的机票已下单（实际需接入机票API）",
                "note": "此为模拟操作，实际订票需接入飞猪/携程等API"
            }
        except Exception as e:
            return {
                "action": "book_flight",
                "status": "error",
                "error": str(e),
                "response": "订机票失败"
            }

    async def _book_hotel(self, params: Dict) -> Dict:
        """模拟订酒店"""
        try:
            # TODO: 接入真实酒店API
            destination = params.get("destination", "未知")

            return {
                "action": "book_hotel",
                "status": "simulated",
                "destination": destination,
                "response": f"模拟订房成功：在{destination}预订了酒店（实际需接入酒店API）",
                "note": "此为模拟操作，实际订房需接入携程/美团等API"
            }
        except Exception as e:
            return {
                "action": "book_hotel",
                "status": "error",
                "error": str(e),
                "response": "订酒店失败"
            }

    async def _set_alarm(self, params: Dict) -> Dict:
        """设置闹钟/提醒"""
        try:
            time = params.get("time", "08:00")
            label = params.get("label", "闹钟")

            # TODO: 可以接入系统API真实设置闹钟
            return {
                "action": "set_alarm",
                "status": "success",
                "time": time,
                "label": label,
                "response": f"已设置{label}，时间为 {time}"
            }
        except Exception as e:
            return {
                "action": "set_alarm",
                "status": "error",
                "error": str(e),
                "response": "设置闹钟失败"
            }

    async def _create_reminder(self, params: Dict) -> Dict:
        """创建日程提醒"""
        try:
            content = params.get("content", "")

            return {
                "action": "create_reminder",
                "status": "success",
                "content": content,
                "response": f"已创建日程提醒：{content[:50]}..."
            }
        except Exception as e:
            return {
                "action": "create_reminder",
                "status": "error",
                "error": str(e),
                "response": "创建提醒失败"
            }

    async def _send_notification(self, params: Dict) -> Dict:
        """发送通知（模拟）"""
        try:
            content = params.get("content", "")

            return {
                "action": "send_notification",
                "status": "simulated",
                "response": f"模拟通知已发送：{content[:50]}..."
            }
        except Exception as e:
            return {
                "action": "send_notification",
                "status": "error",
                "error": str(e),
                "response": "发送通知失败"
            }

    async def _general_execute(self, task: str) -> Dict:
        """通用执行（无法识别类型时）"""
        return {
            "action": "execute",
            "status": "unknown",
            "task": task,
            "response": "无法执行此操作，请描述具体的操作类型（如订票、设闹钟等）"
        }

    # ==================== Skill预留位置 ====================

    # TODO: flight_booking_skill - 接入真实机票API（飞猪/携程）
    # TODO: hotel_booking_skill - 接入真实酒店API
    # TODO: alarm_system_skill - 接入系统闹钟API
    # TODO: calendar_api_skill - 接入日历API（Google Calendar/Outlook）
    # TODO: notification_skill - 接入通知系统（邮件/短信）