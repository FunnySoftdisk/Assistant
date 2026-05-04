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
    """

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

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """处理外部执行请求"""
        if x is None:
            return Msg(name=self.name, content=json.dumps({"error": "No input"}), role="assistant")

        if isinstance(x, list):
            x = x[-1]

        task = x.content if hasattr(x, 'content') else str(x)

        # 解析并执行任务
        result = await self._execute_task(task)
        self.execution_history.append(result)

        return Msg(
            name=self.name,
            content=json.dumps(result, ensure_ascii=False),
            role="assistant"
        )

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
            return await self.builtin_actions[action_type](params)

        # 通用执行
        return await self._general_execute(task)

    async def _book_flight(self, params: Dict) -> Dict:
        """模拟订机票"""
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

    async def _book_hotel(self, params: Dict) -> Dict:
        """模拟订酒店"""
        # TODO: 接入真实酒店API
        destination = params.get("destination", "未知")

        return {
            "action": "book_hotel",
            "status": "simulated",
            "destination": destination,
            "response": f"模拟订房成功：在{destination}预订了酒店（实际需接入酒店API）",
            "note": "此为模拟操作，实际订房需接入携程/美团等API"
        }

    async def _set_alarm(self, params: Dict) -> Dict:
        """设置闹钟/提醒"""
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

    async def _create_reminder(self, params: Dict) -> Dict:
        """创建日程提醒"""
        content = params.get("content", "")

        return {
            "action": "create_reminder",
            "status": "success",
            "content": content,
            "response": f"已创建日程提醒：{content[:50]}..."
        }

    async def _send_notification(self, params: Dict) -> Dict:
        """发送通知（模拟）"""
        content = params.get("content", "")

        return {
            "action": "send_notification",
            "status": "simulated",
            "response": f"模拟通知已发送：{content[:50]}..."
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