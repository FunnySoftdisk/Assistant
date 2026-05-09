"""
信息查询Agent - 外部信息查询
动态从Skill目录发现并调用工具，Skill不可用时降级到DuckDuckGo搜索
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
import json
import asyncio
from typing import Optional, Union, List, Dict

from core.llm_client import llm_chat
from skills.generic_skill import get_generic_skill_loader


class InfoQueryAgent(AgentBase):
    """
    信息查询Agent - 动态Skill调度
    职责：
    1. 动态发现可用的Skills和Tools
    2. 根据查询内容匹配最佳Skill
    3. Skill不可用时降级到DuckDuckGo搜索
    4. 内置时间/日期等基础查询

    特性：
    - 单个Skill失败不影响其他
    - Skill无脚本时自动降级搜索
    - 30秒超时保护
    """

    def __init__(self, name: str = "InfoQueryAgent", model_config: dict = None, **kwargs):
        super().__init__()
        self.name = name
        self.model_config = model_config or {}
        self.memory = InMemoryMemory()

        # 加载Skill加载器
        self.skill_loader = get_generic_skill_loader()
        print(f"✓ InfoQueryAgent 已加载 {len(self.skill_loader.list_skills())} 个Skills")

        # 内置工具（不依赖外部Skill）
        self.builtin_tools = {
            "time": self._query_time,
            "date": self._query_date,
        }

        # 动态生成SYSTEM_PROMPT，包含当前可用的真实Skills
        self.SYSTEM_PROMPT = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """动态构建系统提示，包含当前可用的真实Skills"""
        all_skills = self.skill_loader.list_skills()

        # 收集所有有可执行脚本的tools
        skill_lines = []
        for name, skill in all_skills.items():
            if skill.tools:
                tool_names = [t.name for t in skill.tools if t.script_path]
                if tool_names:
                    tool_str = ", ".join(tool_names)
                    skill_lines.append(f"| {name} | {skill.description[:50]} | {tool_str} |")

        skills_table = "\n".join(skill_lines) if skill_lines else "| 无 | - | - |"

        return f"""你是一个实时信息查询助手，负责回答天气、时间、交通、搜索等问题。

## 你的职责

1. **理解查询意图**：用户是想查天气、查时间、查交通还是搜索信息
2. **匹配Skill**：优先使用已集成的Skill工具
3. **降级搜索**：Skill不可用时使用DuckDuckGo搜索
4. **格式化回答**：将查询结果以友好方式返回

## 查询类型

| 类型 | 关键词示例 | 返回内容 |
|------|-----------|----------|
| 天气 | "北京天气"、"weather 上海" | 温度、天气状况、湿度、风力 |
| 时间 | "现在几点"、"时间" | 当前时间 HH:MM:SS |
| 日期 | "今天几号"、"日期" | 当前日期、工作日 |
| 交通 | "上海路况"、"堵车吗" | 拥堵程度、道路状况 |
| 搜索 | "餐厅评分"、"大众点评"等 | 相关结果摘要 |

## 已集成的Skills（真实可用）

系统已集成以下Skills（共 {len(all_skills)} 个，其中 {sum(1 for s in all_skills.values() if any(t.script_path for t in s.tools))} 个有可执行工具）：

| Skill名称 | 功能说明 | 可用工具 |
|---------|----------|----------|
{skills_table}

## 工具调用流程

1. 先尝试使用匹配的Skill的工具
2. 如果Skill无脚本或调用失败，降级到DuckDuckGo搜索
3. DuckDuckGo搜索是通用的后备方案

## 回答原则

1. **准确**：提供准确的信息，不要编造
2. **简洁**：用简洁语言表达，避免冗长
3. **诚实**：如果搜索也查不到，说明情况
4. **说明来源**：告诉用户数据来自哪个工具/Skill

## 特殊情况

- 如果无法获取天气信息，说明原因并建议用户手动查询
- 如果搜索结果为空，诚实告知并建议其他搜索词
- 如果网络超时，说明暂时无法查询并建议稍后重试"""

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """处理信息查询请求"""
        if x is None:
            return Msg(name=self.name, content=json.dumps({"error": "No input"}), role="assistant")

        if isinstance(x, list):
            x = x[-1]

        query = x.content if hasattr(x, 'content') else str(x)

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
        """检测查询类型和参数"""
        query_lower = query.lower()
        locations = ["北京", "上海", "杭州", "深圳", "广州", "成都", "重庆", "西安", "武汉", "南京", "天津", "苏州"]

        # 时间查询
        if any(kw in query_lower for kw in ["时间", "几点", "now", "clock"]):
            return "time", ""

        # 日期查询
        if any(kw in query_lower for kw in ["日期", "几号", "date", "今天"]):
            return "date", ""

        # 其他查询类型让Skill匹配决定
        return "generic", query

    async def _dispatch_query(self, query: str) -> Dict:
        """分发查询 - 优先内置工具，其次动态匹配Skill，最后DuckDuckGo"""
        query_type, param = self._detect_query_type(query)

        # 内置工具
        if query_type in self.builtin_tools:
            return await self.builtin_tools[query_type](param)

        # 动态匹配Skill
        matched_skills = self.skill_loader.match_skills(query)

        if matched_skills:
            # 尝试按优先级调用匹配的Skill
            for skill in matched_skills:
                result = await self._try_invoke_skill(skill, query)
                if result and "error" not in result:
                    return result

        # 所有Skill都失败或没有匹配的Skill，使用DuckDuckGo搜索
        return await self._search_web(query)

    async def _try_invoke_skill(self, skill, query: str) -> Optional[Dict]:
        """尝试调用Skill的工具"""
        try:
            if not skill.tools:
                # Skill存在但没有定义工具，打印提示用于调试
                print(f"Skill {skill.name} 没有定义工具")
                return None

            # 尝试每个工具
            for tool in skill.tools:
                if not tool.script_path:
                    continue

                # 提取查询参数
                params = self._extract_params(tool.name, query)

                result = await self.skill_loader.invoke_tool(tool, params)
                if result and "error" not in result:
                    result["skill"] = skill.name
                    result["tool"] = tool.name
                    return result

        except Exception as e:
            print(f"Skill {skill.name} 调用失败: {e}")

        return None

    def _extract_params(self, tool_name: str, query: str) -> Dict:
        """从查询中提取工具参数"""
        params = {}

        # 根据工具名称决定提取什么参数
        tool_lower = tool_name.lower()

        if "weather" in tool_lower:
            locations = ["北京", "上海", "杭州", "深圳", "广州", "成都", "重庆", "西安", "武汉", "南京", "天津", "苏州"]
            for loc in locations:
                if loc in query:
                    params["location"] = loc
                    break
            if "location" not in params:
                params["location"] = "北京"

        if "search" in tool_lower or "query" in tool_lower or "dianping" in tool_lower:
            params["keyword"] = query
            params["query"] = query

        if "traffic" in tool_lower or "map" in tool_lower:
            locations = ["北京", "上海", "杭州", "深圳", "广州", "成都", "重庆", "西安", "武汉", "南京", "天津", "苏州"]
            for loc in locations:
                if loc in query:
                    params["location"] = loc
                    break
            if "location" not in params:
                params["location"] = "当前"

        # 默认参数
        if not params:
            params["query"] = query

        return params

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

    async def _search_web(self, query: str) -> Dict:
        """
        DuckDuckGo搜索 - 通用后备方案
        当没有匹配的Skill或Skill调用失败时使用
        """
        try:
            from ddgs import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")[:200]
                    })

            if not results:
                return {
                    "tool": "search",
                    "query": query,
                    "results": [],
                    "response": "未找到相关结果"
                }

            # 生成摘要
            summary_parts = []
            for r in results[:3]:
                title = r.get("title", "")
                snippet = r.get("snippet", "")[:100]
                if title and snippet:
                    summary_parts.append(f"- {title}: {snippet}")

            summary = "\n".join(summary_parts) if summary_parts else results[0].get("snippet", "")

            return {
                "tool": "search",
                "source": "duckduckgo",
                "query": query,
                "results": results,
                "total": len(results),
                "response": f"通过DuckDuckGo搜索「{query}」找到{len(results)}条结果：\n{summary}"
            }

        except ImportError:
            return {
                "tool": "search",
                "query": query,
                "error": "搜索库ddgs未安装",
                "install_hint": "pip install ddgs",
                "response": "搜索功能暂不可用，请运行: pip install ddgs"
            }
        except Exception as e:
            return {
                "tool": "search",
                "query": query,
                "error": str(e),
                "response": f"搜索失败: {str(e)[:50]}"
            }
