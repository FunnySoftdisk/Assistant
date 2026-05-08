"""
Notification Skill - Windows 桌面通知
支持:
1. Windows Toast 通知 (win10toast)
2. Microsoft Todo 集成 (REST API)

Windows通知通过win10toast或plyer实现
Microsoft Todo通过Microsoft Graph API实现（需要OAuth）
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
import asyncio
import sys
import platform
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class NotificationResult:
    """通知结果"""
    success: bool
    notification_type: str
    title: str
    message: str
    error: Optional[str] = None


class NotificationSkill(AgentBase):
    """
    Notification Skill - Windows 桌面通知

    功能：
    1. Windows Toast 通知 (win10toast)
    2. 定时提醒 (基于通知)
    3. Microsoft Todo 操作 (需要认证)

    特性:
    - 跨平台支持（Windows优先）
    - 通知队列管理
    - OAuth认证预留
    """

    def __init__(
        self,
        name: str = "NotificationSkill",
        model_config: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(Name=name, **kwargs)
        self.name = name
        self.model_config = model_config or {}
        self.memory = InMemoryMemory()

        # 通知工具
        self.tools = {
            "toast": self._send_toast,
            "reminder": self._set_reminder,
            "todo": self._microsoft_todo,
        }

        # 平台检测
        self.is_windows = platform.system() == "Windows"

        # Microsoft Todo配置（需要设置Azure AD应用）
        self.ms_graph_config = {
            "client_id": None,  # 需要设置
            "tenant_id": None,  # 需要设置
            "access_token": None,
        }

    async def reply(
        self,
        x: Optional[Union[Msg, List[Msg]]] = None
    ) -> Msg:
        """处理通知请求"""
        if x is None:
            return Msg(
                name=self.name,
                content=json.dumps({"error": "No input provided"}),
                role="assistant"
            )

        if isinstance(x, list):
            x = x[-1]

        query = x.content if hasattr(x, 'content') else str(x)

        # 执行通知（带超时保护）
        try:
            result = await asyncio.wait_for(
                self._execute_notification(query),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            result = {
                "success": False,
                "error": "通知发送超时",
                "notification_type": "unknown"
            }
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "notification_type": "unknown"
            }

        return Msg(
            name=self.name,
            content=json.dumps(result, ensure_ascii=False),
            role="assistant"
        )

    async def _execute_notification(self, query: str) -> Dict[str, Any]:
        """解析并执行通知"""
        query = query.strip()

        # 解析命令
        if ":" in query:
            parts = query.split(":", 1)
            tool_name = parts[0].strip().lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
        else:
            # 智能判断
            tool_name, arg = self._detect_notification_type(query)

        # 执行
        if tool_name in self.tools:
            try:
                result = await self.tools[tool_name](arg)
                return result
            except Exception as e:
                return {
                    "success": False,
                    "error": f"通知失败: {str(e)}",
                    "notification_type": tool_name
                }
        else:
            return {
                "success": False,
                "error": f"未知通知类型: {tool_name}",
                "available_types": list(self.tools.keys())
            }

    def _detect_notification_type(self, query: str) -> tuple:
        """检测通知类型"""
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["通知", "提醒", "notification", "提醒我"]):
            return "toast", query

        if any(kw in query_lower for kw in ["定时", "闹钟", "alarm", "reminder"]):
            return "reminder", query

        if any(kw in query_lower for kw in ["todo", "待办", "任务"]):
            return "todo", query

        # 默认发送通知
        return "toast", query

    async def _send_toast(self, message: str) -> Dict[str, Any]:
        """
        发送 Windows Toast 通知

        支持方式:
        1. win10toast (推荐，Windows 10/11)
        2. plyer (跨平台)
        3. pywin32 (Windows原生)
        """
        if not message:
            return {"success": False, "error": "通知内容不能为空"}

        # 解析标题和内容
        if "|" in message:
            parts = message.split("|", 1)
            title = parts[0].strip()
            content = parts[1].strip()
        else:
            title = "智能助手通知"
            content = message

        # 尝试不同通知库
        result = await self._try_win10toast(title, content)
        if result.get("success"):
            return result

        result = await self._try_plyer(title, content)
        if result.get("success"):
            return result

        # 都失败返回友好提示
        return {
            "success": False,
            "error": "无法发送通知，请安装win10toast: pip install win10toast",
            "notification_type": "toast",
            "title": title,
            "message": content,
            "platform": platform.system()
        }

    async def _try_win10toast(self, title: str, message: str) -> Dict[str, Any]:
        """尝试使用 win10toast"""
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(
                title=title,
                msg=message,
                duration=5,
                threaded=True
            )
            return {
                "success": True,
                "notification_type": "toast",
                "title": title,
                "message": message,
                "method": "win10toast"
            }
        except ImportError:
            return {"success": False, "error": "win10toast not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _try_plyer(self, title: str, message: str) -> Dict[str, Any]:
        """尝试使用 plyer"""
        try:
            from plyer import notification
            notification.notify(
                title=title,
                message=message,
                app_name="Multi-Agent Assistant",
                timeout=5
            )
            return {
                "success": True,
                "notification_type": "toast",
                "title": title,
                "message": message,
                "method": "plyer"
            }
        except ImportError:
            return {"success": False, "error": "plyer not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _set_reminder(self, message: str) -> Dict[str, Any]:
        """
        设置提醒

        输入格式: "标题|时间" 或 "标题"
        时间格式: "15:00" 或 "15:00 2024-01-01"
        """
        if not message:
            return {"success": False, "error": "提醒内容不能为空"}

        # 解析
        parts = message.split("|")
        title = parts[0].strip()
        time_str = parts[1].strip() if len(parts) > 1 else None

        # 解析时间
        reminder_time = None
        if time_str:
            try:
                if " " in time_str:
                    reminder_time = datetime.strptime(time_str, "%H:%M %Y-%m-%d")
                else:
                    # 今天的这个时间
                    today = datetime.now().strftime("%Y-%m-%d")
                    reminder_time = datetime.strptime(f"{today} {time_str}", "%Y-%m-%d %H:%M")
            except ValueError:
                return {
                    "success": False,
                    "error": f"时间格式不正确: {time_str}，请使用 HH:MM 或 HH:MM YYYY-MM-DD"
                }
        else:
            # 默认30秒后提醒（仅演示）
            reminder_time = datetime.now() + timedelta(seconds=30)

        # 异步等待并发送提醒
        asyncio.create_task(self._delayed_reminder(title, reminder_time))

        return {
            "success": True,
            "notification_type": "reminder",
            "title": title,
            "reminder_time": reminder_time.isoformat(),
            "message": f"已设置提醒: {title}",
            "note": "提醒将在指定时间发送通知"
        }

    async def _delayed_reminder(self, title: str, reminder_time: datetime):
        """延迟发送提醒"""
        # 计算等待时间
        now = datetime.now()
        wait_seconds = (reminder_time - now).total_seconds()

        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        # 发送通知
        await self._send_toast(f"提醒: {title}")

    async def _microsoft_todo(self, action: str) -> Dict[str, Any]:
        """
        Microsoft Todo 操作

        支持操作:
        - todo:add:任务内容 - 添加待办
        - todo:list - 列出所有待办
        - todo:complete:任务ID - 完成任务
        - todo:delete:任务ID - 删除待办

        注意: 需要配置 Azure AD 应用和 OAuth 认证
        """
        if not self.ms_graph_config.get("access_token"):
            return {
                "success": False,
                "error": "Microsoft Todo 需要先配置 OAuth 认证",
                "notification_type": "todo",
                "setup_hint": """
                请按以下步骤配置:
                1. 在 Azure Portal 创建 Azure AD 应用
                2. 配置应用权限: Tasks.ReadWrite
                3. 获取 client_id 和 tenant_id
                4. 实现 OAuth 2.0 认证流程
                """,
                "alternative": "可使用本地通知作为替代: toast:标题|内容"
            }

        # 解析操作
        if ":" in action:
            action_type, content = action.split(":", 1)
        else:
            action_type = action
            content = ""

        try:
            if action_type == "add":
                return await self._ms_todo_add(content)
            elif action_type == "list":
                return await self._ms_todo_list()
            elif action_type == "complete":
                return await self._ms_todo_complete(content)
            elif action_type == "delete":
                return await self._ms_todo_delete(content)
            else:
                return {
                    "success": False,
                    "error": f"未知操作: {action_type}",
                    "available_actions": ["add", "list", "complete", "delete"]
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Microsoft Todo 操作失败: {str(e)}"
            }

    async def _ms_todo_add(self, content: str) -> Dict[str, Any]:
        """添加 Microsoft Todo 任务"""
        if not content:
            return {"success": False, "error": "任务内容不能为空"}

        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.ms_graph_config['access_token']}",
                "Content-Type": "application/json"
            }

            # 获取默认任务列表
            lists_response = await httpx.AsyncClient().get(
                "https://graph.microsoft.com/v1.0/me/todo/lists",
                headers=headers
            )

            if lists_response.status_code != 200:
                return {
                    "success": False,
                    "error": f"获取任务列表失败: {lists_response.status_code}"
                }

            lists = lists_response.json().get("value", [])
            if not lists:
                return {
                    "success": False,
                    "error": "未找到任何任务列表"
                }

            # 添加任务到第一个列表
            list_id = lists[0].get("id")
            task_data = {
                "title": content,
                "body": {
                    "contentType": "text",
                    "content": f"创建时间: {datetime.now().isoformat()}"
                }
            }

            task_response = await httpx.AsyncClient().post(
                f"https://graph.microsoft.com/v1.0/me/todo/lists/{list_id}/tasks",
                headers=headers,
                json=task_data
            )

            if task_response.status_code == 201:
                task = task_response.json()
                return {
                    "success": True,
                    "notification_type": "todo",
                    "action": "add",
                    "task_id": task.get("id"),
                    "title": content,
                    "message": f"已添加到 Microsoft Todo"
                }
            else:
                return {
                    "success": False,
                    "error": f"添加任务失败: {task_response.status_code}"
                }

        except ImportError:
            return {"success": False, "error": "需要安装 httpx: pip install httpx"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _ms_todo_list(self) -> Dict[str, Any]:
        """列出 Microsoft Todo 任务"""
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.ms_graph_config['access_token']}",
                "Content-Type": "application/json"
            }

            # 获取任务列表
            lists_response = await httpx.AsyncClient().get(
                "https://graph.microsoft.com/v1.0/me/todo/lists",
                headers=headers
            )

            if lists_response.status_code != 200:
                return {
                    "success": False,
                    "error": f"获取任务列表失败: {lists_response.status_code}"
                }

            lists = lists_response.json().get("value", [])
            all_tasks = []

            # 获取每个列表的任务
            for lst in lists[:3]:  # 限制最多3个列表
                list_id = lst.get("id")
                tasks_response = await httpx.AsyncClient().get(
                    f"https://graph.microsoft.com/v1.0/me/todo/lists/{list_id}/tasks",
                    headers=headers,
                    params={"$filter": "status ne 'completed'"}
                )

                if tasks_response.status_code == 200:
                    tasks = tasks_response.json().get("value", [])
                    for task in tasks[:10]:  # 每个列表最多10个
                        all_tasks.append({
                            "id": task.get("id"),
                            "title": task.get("title"),
                            "list": lst.get("displayName"),
                            "status": task.get("status")
                        })

            return {
                "success": True,
                "notification_type": "todo",
                "action": "list",
                "tasks": all_tasks,
                "total": len(all_tasks)
            }

        except ImportError:
            return {"success": False, "error": "需要安装 httpx"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _ms_todo_complete(self, task_id: str) -> Dict[str, Any]:
        """完成任务"""
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.ms_graph_config['access_token']}",
                "Content-Type": "application/json"
            }

            # 查找任务所在的列表
            lists_response = await httpx.AsyncClient().get(
                "https://graph.microsoft.com/v1.0/me/todo/lists",
                headers=headers
            )

            for lst in lists_response.json().get("value", []):
                list_id = lst.get("id")
                task_response = await httpx.AsyncClient().get(
                    f"https://graph.microsoft.com/v1.0/me/todo/lists/{list_id}/tasks/{task_id}",
                    headers=headers
                )

                if task_response.status_code == 200:
                    # 更新任务状态
                    update_response = await httpx.AsyncClient().patch(
                        f"https://graph.microsoft.com/v1.0/me/todo/lists/{list_id}/tasks/{task_id}",
                        headers=headers,
                        json={"status": "completed"}
                    )

                    if update_response.status_code == 200:
                        return {
                            "success": True,
                            "notification_type": "todo",
                            "action": "complete",
                            "task_id": task_id,
                            "message": "任务已完成"
                        }

            return {
                "success": False,
                "error": f"未找到任务: {task_id}"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _ms_todo_delete(self, task_id: str) -> Dict[str, Any]:
        """删除任务"""
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.ms_graph_config['access_token']}"
            }

            # 查找并删除
            lists_response = await httpx.AsyncClient().get(
                "https://graph.microsoft.com/v1.0/me/todo/lists",
                headers=headers
            )

            for lst in lists_response.json().get("value", []):
                list_id = lst.get("id")
                delete_response = await httpx.AsyncClient().delete(
                    f"https://graph.microsoft.com/v1.0/me/todo/lists/{list_id}/tasks/{task_id}",
                    headers=headers
                )

                if delete_response.status_code == 204:
                    return {
                        "success": True,
                        "notification_type": "todo",
                        "action": "delete",
                        "task_id": task_id,
                        "message": "任务已删除"
                    }

            return {
                "success": False,
                "error": f"未找到任务: {task_id}"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_microsoft_token(self, access_token: str):
        """设置 Microsoft Graph 访问令牌"""
        self.ms_graph_config["access_token"] = access_token

    def configure_microsoft(self, client_id: str, tenant_id: str):
        """配置 Microsoft Todo OAuth"""
        self.ms_graph_config["client_id"] = client_id
        self.ms_graph_config["tenant_id"] = tenant_id


# Skill 入口函数
def create_skill_agent(model_config: dict = None) -> NotificationSkill:
    """创建 NotificationSkill Agent 实例"""
    return NotificationSkill(model_config=model_config)
