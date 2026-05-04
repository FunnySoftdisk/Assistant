"""
信息查询Agent - 外部信息查询
负责天气、时间、交通等实时信息查询
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
from typing import Optional, Union, List, Dict

from core.llm_client import llm_chat


class InfoQueryAgent(AgentBase):
    """
    信息查询Agent - 外部信息查询
    职责：
    1. 天气查询 - 调用天气API
    2. 时间查询 - 获取当前日期时间
    3. 交通查询 - 公交、地铁、路况等
    4. 通用搜索 - DuckDuckGo联网搜索
    """

    def __init__(self, name: str = "InfoQueryAgent", model_config: dict = None, **kwargs):
        super().__init__()
        self.name = name
        self.model_config = model_config or {}
        self.memory = InMemoryMemory()

        # 内置查询工具
        self.builtin_tools = {
            "weather": self._query_weather,
            "time": self._query_time,
            "date": self._query_date,
            "traffic": self._query_traffic,
        }

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """处理信息查询请求"""
        if x is None:
            return Msg(name=self.name, content=json.dumps({"error": "No input"}), role="assistant")

        if isinstance(x, list):
            x = x[-1]

        query = x.content if hasattr(x, 'content') else str(x)

        # 检测查询类型并执行
        result = await self._dispatch_query(query)

        return Msg(
            name=self.name,
            content=json.dumps(result, ensure_ascii=False),
            role="assistant"
        )

    def _detect_query_type(self, query: str) -> tuple:
        """检测查询类型"""
        query_lower = query.lower()

        # 天气查询
        if any(kw in query_lower for kw in ["天气", "weather", "温度", "气温"]):
            location = query_lower.replace("天气", "").replace("weather", "").replace("温度", "").replace("气温", "").strip()
            return "weather", location or "北京"

        # 时间查询
        if any(kw in query_lower for kw in ["时间", "几点", "now", "clock"]):
            return "time", ""

        # 日期查询
        if any(kw in query_lower for kw in ["日期", "几号", "date", "今天"]):
            return "date", ""

        # 交通查询
        if any(kw in query_lower for kw in ["交通", "路况", "堵车", "traffic"]):
            location = query_lower.replace("交通", "").replace("路况", "").replace("堵车", "").strip()
            return "traffic", location or "当前"

        # 通用搜索
        return "search", query

    async def _dispatch_query(self, query: str) -> Dict:
        """分发查询到对应工具"""
        query_type, param = self._detect_query_type(query)

        if query_type in self.builtin_tools:
            return await self.builtin_tools[query_type](param)

        # 默认执行搜索
        return await self._search_web(query)

    async def _query_weather(self, location: str) -> Dict:
        """
        查询天气
        实际项目中应接入真实天气API
        """
        try:
            # TODO: 接入真实天气API（如和风天气、OpenWeatherMap等）
            # 这里使用模拟数据作为占位
            from datetime import datetime
            hour = datetime.now().hour

            # 简单模拟天气
            weather_conditions = ["晴", "多云", "阴", "小雨"]
            condition = weather_conditions[hour % 4]
            temp = 20 + (hour % 10)

            result = {
                "tool": "weather",
                "location": location,
                "temperature": f"{temp}°C",
                "condition": condition,
                "humidity": "55%",
                "wind": "东南风 2级",
                "response": f"{location}今天天气{condition}，{temp}°C，湿度55%，适合出行"
            }

            # 如果有搜索库，尝试真实搜索
            try:
                result_from_search = await self._search_web(f"{location}今天天气")
                if "error" not in result_from_search:
                    result = result_from_search
            except:
                pass

            return result

        except Exception as e:
            return {
                "tool": "weather",
                "error": str(e),
                "response": f"查询天气失败: {str(e)}"
            }

    async def _query_time(self, _: str) -> Dict:
        """查询当前时间"""
        from datetime import datetime
        now = datetime.now()
        return {
            "tool": "time",
            "time": now.strftime("%H:%M:%S"),
            "response": f"现在是 {now.strftime('%H:%M:%S')}"
        }

    async def _query_date(self, _: str) -> Dict:
        """查询当前日期"""
        from datetime import datetime
        now = datetime.now()
        return {
            "tool": "date",
            "date": now.strftime("%Y年%m月%d日"),
            "weekday": now.strftime("%A"),
            "response": f"今天是 {now.strftime('%Y年%m月%d日')}，{now.strftime('%A')}"
        }

    async def _query_traffic(self, location: str) -> Dict:
        """
        查询交通/路况
        实际项目中应接入地图API（如高德、百度）
        """
        # TODO: 接入真实地图API
        return {
            "tool": "traffic",
            "location": location,
            "status": "畅通",
            "congestion_level": "低",
            "response": f"{location}当前交通状况良好，道路畅通"
        }

    async def _search_web(self, query: str) -> Dict:
        """
        执行网络搜索
        使用DuckDuckGo
        """
        try:
            from duckduckgo_search import AsyncDDGS

            results = []
            async with AsyncDDGS() as ddgs:
                async for r in ddgs.async_aiterator(query, max_results=3):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("body", "")[:150]
                    })

            if not results:
                return {
                    "tool": "search",
                    "query": query,
                    "results": [],
                    "response": "未找到相关结果"
                }

            # 生成摘要
            summary = results[0].get("snippet", "")
            if len(results) > 1:
                summary = f"找到{len(results)}条结果：{results[0]['snippet']}"

            return {
                "tool": "search",
                "query": query,
                "results": results,
                "total": len(results),
                "response": summary
            }

        except ImportError:
            return {
                "tool": "search",
                "query": query,
                "error": "搜索功能需要安装ddgs库",
                "install_hint": "pip install ddgs",
                "response": "搜索功能暂不可用"
            }
        except Exception as e:
            return {
                "tool": "search",
                "query": query,
                "error": str(e),
                "response": f"搜索失败: {str(e)}"
            }

    # ==================== Skill预留位置 ====================

    # TODO: weather_api_skill - 接入真实天气API
    # TODO: map_api_skill - 接入地图API（路况、公交）
    # TODO: news_skill - 新闻查询
    # TODO: stock_skill - 股票查询