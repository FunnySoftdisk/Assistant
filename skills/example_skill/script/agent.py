"""
Tool Skill - 真实可用的网络搜索工具
使用DuckDuckGo API进行网络搜索
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
import asyncio
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass


@dataclass
class SearchResult:
    """搜索结果"""
    title: str
    url: str
    snippet: str


class ToolSkill(AgentBase):
    """
    Tool Skill - 工具执行Agent

    功能：
    1. 网络搜索 (DuckDuckGo)
    2. 天气查询
    3. 计算器
    4. 日期/时间查询

    这是一个完整可用的Skill，展示了如何实现真实的工具调用
    """

    def __init__(
        self,
        name: str = "ToolSkill",
        model_config: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.model_config = model_config or {}
        self.memory = InMemoryMemory()

        self.tools = {
            "search": self._search_web,
            "weather": self._get_weather,
            "calc": self._calculate,
            "time": self._get_time,
            "date": self._get_date,
        }

    async def reply(
        self,
        x: Optional[Union[Msg, List[Msg]]] = None
    ) -> Msg:
        """
        处理工具调用请求
        """
        if x is None:
            return Msg(
                name=self.name,
                content=json.dumps({"error": "No input provided"}),
                role="assistant"
            )

        if isinstance(x, list):
            x = x[-1]

        query = x.content if hasattr(x, 'content') else str(x)

        # 解析工具调用
        result = await self._execute_tool(query)

        return Msg(
            name=self.name,
            content=json.dumps(result, ensure_ascii=False),
            role="assistant"
        )

    async def _execute_tool(self, query: str) -> Dict[str, Any]:
        """
        解析并执行工具

        输入格式: "tool_name:参数" 或 "search:天气"
        """
        query = query.strip()

        # 解析命令
        if ":" in query:
            tool_name, arg = query.split(":", 1)
            tool_name = tool_name.strip()
            arg = arg.strip()
        else:
            # 智能判断工具类型
            tool_name, arg = self._detect_tool(query)

        # 执行工具
        if tool_name in self.tools:
            try:
                result = await self.tools[tool_name](arg)
                return result
            except Exception as e:
                return {"error": f"Tool execution failed: {str(e)}", "tool": tool_name}
        else:
            return {"error": f"Unknown tool: {tool_name}", "available_tools": list(self.tools.keys())}

    def _detect_tool(self, query: str) -> tuple:
        """根据查询内容智能判断工具类型"""
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["搜索", "search", "查找", "查询"]):
            keyword = query.replace("搜索", "").replace("search", "").replace("查找", "").replace("查询", "").strip()
            return "search", keyword

        if any(kw in query_lower for kw in ["天气", "weather", "温度"]):
            location = query_lower.replace("天气", "").replace("weather", "").replace("温度", "").strip()
            return "weather", location or "北京"

        if any(kw in query_lower for kw in ["计算", "calc", "等于"]):
            expr = query_lower.replace("计算", "").replace("calc", "").replace("等于", "").strip()
            return "calc", expr

        if any(kw in query_lower for kw in ["时间", "现在几点", "time"]):
            return "time", ""

        if any(kw in query_lower for kw in ["日期", "今天几号", "date"]):
            return "date", ""

        # 默认使用搜索
        return "search", query

    async def _search_web(self, query: str) -> Dict[str, Any]:
        """
        执行网络搜索 - 使用真实的DuckDuckGo API

        使用 ddgs 库 (DuckDuckGo Search)
        安装: pip install ddgs
        """
        try:
            from duckduckgo_search import AsyncDDGS

            results = []
            async with AsyncDDGS() as ddgs:
                async for r in ddgs.async_aiterator(query, max_results=5):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("body", "")[:200]
                    })

            return {
                "tool": "search",
                "query": query,
                "results": results,
                "total": len(results)
            }

        except ImportError:
            # ddgs未安装，返回友好的错误信息
            return {
                "tool": "search",
                "query": query,
                "results": [],
                "total": 0,
                "error": "搜索功能需要安装ddgs库",
                "install_hint": "pip install ddgs"
            }
        except Exception as e:
            return {
                "tool": "search",
                "query": query,
                "results": [],
                "error": f"Search failed: {str(e)}"
            }

    async def _get_weather(self, location: str) -> Dict[str, Any]:
        """获取天气信息（模拟）"""
        # 实际项目中可以接入天气API
        return {
            "tool": "weather",
            "location": location,
            "temperature": "25°C",
            "condition": "多云",
            "humidity": "60%",
            "wind": "东南风 3级",
            "note": "模拟数据，实际项目请接入真实天气API"
        }

    async def _calculate(self, expression: str) -> Dict[str, Any]:
        """执行计算"""
        try:
            # 安全计算（仅支持基本运算）
            allowed_chars = set("0123456789+-*/.() ")
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return {
                    "tool": "calc",
                    "expression": expression,
                    "result": result
                }
            else:
                return {
                    "tool": "calc",
                    "error": "表达式包含非法字符"
                }
        except Exception as e:
            return {
                "tool": "calc",
                "expression": expression,
                "error": f"Calculation error: {str(e)}"
            }

    async def _get_time(self, _: str) -> Dict[str, Any]:
        """获取当前时间"""
        from datetime import datetime
        now = datetime.now()
        return {
            "tool": "time",
            "time": now.strftime("%H:%M:%S"),
            "datetime": now.isoformat()
        }

    async def _get_date(self, _: str) -> Dict[str, Any]:
        """获取当前日期"""
        from datetime import datetime
        now = datetime.now()
        return {
            "tool": "date",
            "date": now.strftime("%Y-%m-%d"),
            "weekday": now.strftime("%A"),
            "datetime": now.isoformat()
        }


# Skill入口函数
def create_skill_agent(model_config: dict = None) -> ToolSkill:
    """创建ToolSkill Agent实例"""
    return ToolSkill(model_config=model_config)