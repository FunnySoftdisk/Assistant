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
    2. 天气查询（模拟）
    3. 计算器
    4. 日期/时间查询
    5. 单位转换
    6. 翻译（模拟）

    特性:
    - 异步执行
    - 超时保护（10秒）
    - 优雅错误处理
    - 网络失败回退
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
            "unit_convert": self._unit_convert,
            "translate": self._translate,
        }

        # 工具超时
        self.tool_timeout = 10.0

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

        # 解析工具调用（带超时保护）
        try:
            result = await asyncio.wait_for(
                self._execute_tool(query),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            result = {"error": "工具执行超时", "tool": "unknown"}
        except Exception as e:
            result = {"error": f"执行失败: {str(e)}", "tool": "unknown"}

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

        # 执行工具（带超时保护）
        if tool_name in self.tools:
            try:
                result = await asyncio.wait_for(
                    self.tools[tool_name](arg),
                    timeout=self.tool_timeout
                )
                return result
            except asyncio.TimeoutError:
                return {"error": f"{tool_name} 执行超时", "tool": tool_name}
            except Exception as e:
                return {"error": f"Tool execution failed: {str(e)}", "tool": tool_name}
        else:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys())
            }

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

        if any(kw in query_lower for kw in ["转换", "换算", "convert"]):
            return "unit_convert", query

        if any(kw in query_lower for kw in ["翻译", "translate"]):
            return "translate", query

        # 默认使用搜索
        return "search", query

    async def _search_web(self, query: str) -> Dict[str, Any]:
        """
        执行网络搜索 - 使用真实的DuckDuckGo API
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

            if not results:
                return {
                    "tool": "search",
                    "query": query,
                    "results": [],
                    "total": 0,
                    "response": "未找到相关结果"
                }

            return {
                "tool": "search",
                "query": query,
                "results": results,
                "total": len(results),
                "response": f"找到{len(results)}条结果：{results[0].get('snippet', '')}"
            }

        except ImportError:
            return {
                "tool": "search",
                "query": query,
                "results": [],
                "total": 0,
                "error": "搜索功能需要安装ddgs库",
                "install_hint": "pip install ddgs",
                "response": "搜索功能暂不可用"
            }
        except asyncio.TimeoutError:
            return {
                "tool": "search",
                "query": query,
                "error": "搜索超时",
                "response": "搜索超时，请稍后重试"
            }
        except Exception as e:
            return {
                "tool": "search",
                "query": query,
                "results": [],
                "error": str(e),
                "response": f"搜索失败: {str(e)[:50]}"
            }

    async def _get_weather(self, location: str) -> Dict[str, Any]:
        """
        获取天气信息（模拟）
        实际项目中应接入真实天气API（如和风天气API）
        """
        try:
            from datetime import datetime
            hour = datetime.now().hour

            weather_conditions = ["晴", "多云", "阴", "小雨", "中雨", "雷阵雨"]
            condition = weather_conditions[hour % len(weather_conditions)]
            temp = 18 + (hour % 10)

            return {
                "tool": "weather",
                "location": location or "北京",
                "temperature": f"{temp}°C",
                "condition": condition,
                "humidity": f"{50 + (hour % 30)}%",
                "wind": "东南风 2-3级",
                "response": f"{location}今天天气{condition}，{temp}°C，适合出行",
                "note": "模拟数据，实际项目请接入天气API"
            }
        except Exception as e:
            return {
                "tool": "weather",
                "error": str(e),
                "response": "查询天气失败"
            }

    async def _calculate(self, expression: str) -> Dict[str, Any]:
        """执行计算（安全）"""
        try:
            # 安全计算（仅支持基本运算）
            allowed_chars = set("0123456789+-*/.() ")
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return {
                    "tool": "calc",
                    "expression": expression,
                    "result": result,
                    "response": f"{expression} = {result}"
                }
            else:
                return {
                    "tool": "calc",
                    "error": "表达式包含非法字符",
                    "response": "不支持的表达式"
                }
        except Exception as e:
            return {
                "tool": "calc",
                "expression": expression,
                "error": f"计算错误: {str(e)}",
                "response": "计算失败"
            }

    async def _get_time(self, _: str) -> Dict[str, Any]:
        """获取当前时间"""
        try:
            from datetime import datetime
            now = datetime.now()
            return {
                "tool": "time",
                "time": now.strftime("%H:%M:%S"),
                "datetime": now.isoformat(),
                "response": f"现在是 {now.strftime('%H:%M:%S')}"
            }
        except Exception as e:
            return {
                "tool": "time",
                "error": str(e),
                "response": "查询时间失败"
            }

    async def _get_date(self, _: str) -> Dict[str, Any]:
        """获取当前日期"""
        try:
            from datetime import datetime
            now = datetime.now()
            weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            weekday = weekdays[now.weekday()]
            return {
                "tool": "date",
                "date": now.strftime("%Y年%m月%d日"),
                "weekday": weekday,
                "datetime": now.isoformat(),
                "response": f"今天是 {now.strftime('%Y年%m月%d日')}，{weekday}"
            }
        except Exception as e:
            return {
                "tool": "date",
                "error": str(e),
                "response": "查询日期失败"
            }

    async def _unit_convert(self, query: str) -> Dict[str, Any]:
        """单位转换"""
        try:
            # 简单实现，实际可用更复杂的库
            conversions = {
                ("公里", "英里"): 0.621371,
                ("英里", "公里"): 1.60934,
                ("公里", "米"): 1000,
                ("米", "厘米"): 100,
                ("千克", "磅"): 2.20462,
                ("磅", "千克"): 0.453592,
                ("摄氏度", "华氏度"): lambda c: c * 9/5 + 32,
                ("华氏度", "摄氏度"): lambda f: (f - 32) * 5/9,
            }

            for (from_unit, to_unit), factor in conversions.items():
                if from_unit in query and to_unit in query:
                    import re
                    numbers = re.findall(r'\d+\.?\d*', query)
                    if numbers:
                        num = float(numbers[0])
                        if callable(factor):
                            result = factor(num)
                        else:
                            result = num * factor
                        return {
                            "tool": "unit_convert",
                            "from": f"{num} {from_unit}",
                            "to": f"{result:.2f} {to_unit}",
                            "response": f"{num} {from_unit} = {result:.2f} {to_unit}"
                        }

            return {
                "tool": "unit_convert",
                "error": "不支持的转换",
                "response": "支持的转换：公里/英里、米/厘米、千克/磅、摄氏度/华氏度"
            }
        except Exception as e:
            return {
                "tool": "unit_convert",
                "error": str(e),
                "response": "单位转换失败"
            }

    async def _translate(self, query: str) -> Dict[str, Any]:
        """翻译（模拟）"""
        # TODO: 接入真实翻译API（百度翻译、腾讯翻译等）
        return {
            "tool": "translate",
            "query": query,
            "response": f"翻译功能模拟：{query}（实际需接入翻译API）",
            "note": "翻译API预留位置"
        }


# Skill入口函数
def create_skill_agent(model_config: dict = None) -> ToolSkill:
    """创建ToolSkill Agent实例"""
    return ToolSkill(model_config=model_config)