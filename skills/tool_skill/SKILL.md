# Tool Skill 配置

name: tool_skill
version: 1.0.0
description: 工具Skill - 提供网络搜索、天气查询、计算器、日期时间、单位转换等实用工具
agent_type: info_query
priority: 1
tools: ["search", "weather", "calc", "time", "date", "unit_convert", "translate"]
parameters: {"query": "工具名称:参数，如 search:天气", "max_results": 5}