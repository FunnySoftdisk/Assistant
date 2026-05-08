"""
信息查询Agent - 外部信息查询
负责天气、时间、交通等实时信息查询
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
import asyncio
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

    特性:
    - 搜索结果缓存
    - 网络超时保护（10秒）
    - 搜索库缺失时的友好提示
    - 单个查询失败不影响其他
    """

    SYSTEM_PROMPT = """你是一个实时信息查询助手，负责回答天气、时间、交通、搜索等问题。

## 你的职责

1. **理解查询意图**：用户是想查天气、查时间、查交通还是搜索信息
2. **提取查询参数**：地点、时间、关键词等
3. **执行查询**：调用相应工具获取信息
4. **格式化回答**：将查询结果以友好方式返回

## 查询类型

| 类型 | 关键词示例 | 返回内容 |
|------|-----------|----------|
| 天气 | "北京天气"、"weather 上海" | 温度、天气状况、湿度、风力 |
| 时间 | "现在几点"、"时间" | 当前时间 HH:MM:SS |
| 日期 | "今天几号"、"日期" | 当前日期、工作日 |
| 交通 | "上海路况"、"堵车吗" | 拥堵程度、道路状况 |
| 搜索 | "搜索xxx"、"查一下xxx" | 相关结果摘要 |

## 工具调用

你会通过以下内置工具获取信息：
- weather: 查询指定地点的天气
- time: 获取当前时间
- date: 获取当前日期
- traffic: 查询交通状况
- search: 执行网络搜索

## 回答原则

1. **准确**：提供准确的信息，不要编造
2. **简洁**：用简洁语言表达，避免冗长
3. **有用**：如果信息不完整，说明情况并给出建议
4. **友好**：使用友好的语气

## 特殊情况

- 如果无法获取天气信息，说明原因并建议用户手动查询
- 如果搜索结果为空，诚实告知并建议其他搜索词
- 如果网络超时，说明暂时无法查询并建议稍后重试"""

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

        # 搜索超时配置
        self.search_timeout = 10.0

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """处理信息查询请求"""
        if x is None:
            return Msg(name=self.name, content=json.dumps({"error": "No input"}), role="assistant")

        if isinstance(x, list):
            x = x[-1]

        query = x.content if hasattr(x, 'content') else str(x)

        # 检测查询类型并执行
        try:
            result = await asyncio.wait_for(
                self._dispatch_query(query),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            result = {
                "tool": "unknown",
                "error": "查询超时",
                "response": "信息查询超时，请稍后重试"
            }
        except Exception as e:
            result = {
                "tool": "unknown",
                "error": str(e),
                "response": f"查询失败: {str(e)[:50]}"
            }

        return Msg(
            name=self.name,
            content=json.dumps(result, ensure_ascii=False),
            role="assistant"
        )

    def _detect_query_type(self, query: str) -> tuple:
        """检测查询类型"""
        query_lower = query.lower()

        # 地点列表
        locations = ["北京", "上海", "杭州", "深圳", "广州", "成都", "重庆", "西安", "武汉", "南京", "天津", "苏州"]

        # 天气查询
        if any(kw in query_lower for kw in ["天气", "weather", "温度", "气温"]):
            # 先提取地点
            found_location = None
            for loc in locations:
                if loc in query:
                    found_location = loc
                    break
            # 如果没找到具体地点，返回默认位置
            return "weather", found_location or "北京"

        # 时间查询
        if any(kw in query_lower for kw in ["时间", "几点", "now", "clock"]):
            return "time", ""

        # 日期查询
        if any(kw in query_lower for kw in ["日期", "几号", "date", "今天"]):
            return "date", ""

        # 交通查询
        if any(kw in query_lower for kw in ["交通", "路况", "堵车", "traffic"]):
            found_location = None
            for loc in locations:
                if loc in query:
                    found_location = loc
                    break
            return "traffic", found_location or "当前"

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

            # 尝试真实搜索（带超时保护）
            try:
                result_from_search = await asyncio.wait_for(
                    self._search_web(f"{location}今天天气"),
                    timeout=self.search_timeout
                )
                if "error" not in result_from_search and result_from_search.get("response"):
                    result = result_from_search
            except (asyncio.TimeoutError, Exception):
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
        try:
            from datetime import datetime
            now = datetime.now()
            return {
                "tool": "time",
                "time": now.strftime("%H:%M:%S"),
                "response": f"现在是 {now.strftime('%H:%M:%S')}"
            }
        except Exception as e:
            return {
                "tool": "time",
                "error": str(e),
                "response": "查询时间失败"
            }

    async def _query_date(self, _: str) -> Dict:
        """查询当前日期"""
        try:
            from datetime import datetime
            now = datetime.now()
            return {
                "tool": "date",
                "date": now.strftime("%Y年%m月%d日"),
                "weekday": now.strftime("%A"),
                "response": f"今天是 {now.strftime('%Y年%m月%d日')}，{now.strftime('%A')}"
            }
        except Exception as e:
            return {
                "tool": "date",
                "error": str(e),
                "response": "查询日期失败"
            }

    async def _query_traffic(self, location: str) -> Dict:
        """
        查询交通/路况
        实际项目中应接入地图API（如高德、百度）
        """
        # TODO: 接入真实地图API
        try:
            return {
                "tool": "traffic",
                "location": location,
                "status": "畅通",
                "congestion_level": "低",
                "response": f"{location}当前交通状况良好，道路畅通"
            }
        except Exception as e:
            return {
                "tool": "traffic",
                "error": str(e),
                "response": f"查询交通失败: {str(e)}"
            }

    async def _search_web(self, query: str) -> Dict:
        """
        执行网络搜索
        使用DuckDuckGo
        """
        try:
            # 检查是否有ddgs库
            try:
                from duckduckgo_search import AsyncDDGS
            except ImportError:
                return {
                    "tool": "search",
                    "query": query,
                    "error": "搜索功能需要安装ddgs库",
                    "install_hint": "pip install ddgs",
                    "response": "搜索功能暂不可用，如需启用请运行: pip install ddgs"
                }

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

        except asyncio.TimeoutError:
            return {
                "tool": "search",
                "query": query,
                "error": "搜索超时",
                "response": "搜索超时，请稍后重试"
            }
        except ImportError:
            return {
                "tool": "search",
                "query": query,
                "error": "搜索库未安装",
                "response": "搜索功能暂不可用"
            }
        except Exception as e:
            return {
                "tool": "search",
                "query": query,
                "error": str(e),
                "response": f"搜索失败: {str(e)[:50]}"
            }

    # ==================== Skill预留位置 ====================

    # TODO: weather_api_skill - 接入真实天气API
    # TODO: map_api_skill - 接入地图API（路况、公交）
    # TODO: news_skill - 新闻查询
    # TODO: stock_skill - 股票查询