# Multi-Agent Assistant

基于AgentScope框架的多Agent智能助手系统，实现意图识别、调度、偏好查询、信息查询、外部执行、日程规划、总结对话、记忆更新等功能。采用"优先级+并行"混合调度模式，FastAPI后端 + 对话式前端。

---

## 架构总览

```
用户消息 → IntentionAgent → OrchestrationAgent
                                    ↓
              ┌─ Priority 1 (并行) ──────────┐
              │  memory_agent (查用户偏好)      │
              │  info_query_agent (查天气/交通) │
              └───────────────────────────────┘
                                    ↓ (等待P1完成)
              ┌─ Priority 2 (依赖P1) ────────┐
              │  planning_agent (生成行程)     │
              └───────────────────────────────┘
                                    ↓ (等待P2完成)
              ┌─ Priority 3 (依赖P2) ────────┐
              │  execution_agent (执行操作)    │
              └───────────────────────────────┘
                                    ↓
              ┌─ Summarization (整合结果) ────┐
              │  summarization_agent (总结)    │
              └───────────────────────────────┘
                                    ↓
                              返回结果给用户
```

---

## 核心Agent (agents/)

| Agent | 职责 | 调度优先级 |
|-------|------|----------|
| `intention_agent.py` | 语义理解，意图识别 | 初始阶段 |
| `orchestration_agent.py` | 调度中心，优先级+并行分发 | 核心编排 |
| `memory_agent.py` | 记忆存储/更新，偏好匹配 | P1 (并行) |
| `info_query_agent.py` | 天气、时间、交通查询 | P1 (并行) |
| `planning_agent.py` | 行程规划生成 | P2 (依赖P1) |
| `execution_agent.py` | 订票、定闹钟等外部操作 | P3 (依赖P2) |
| `summarization_agent.py` | 对话总结，整合结果告诉用户 | Summarize (最后) |
| `preference_agent.py` | 偏好查询/管理 | P1 (并行) |

---

## 调度模式详解

### 优先级定义

| 优先级 | Agent | 依赖关系 | 说明 |
|--------|-------|----------|------|
| **P1** | memory_agent, info_query_agent | 无 | 并行执行，快速获取用户偏好和外部信息 |
| **P2** | planning_agent | 依赖P1 | 等P1完成后生成行程规划 |
| **P3** | execution_agent | 依赖P2 | 等P2生成规划后执行具体操作 |
| **Summarize** | summarization_agent | 依赖P3 | 整合所有结果，返回给用户 |

### 意图到调度映射

```python
intent_agent_map = {
    "travel_planning": {
        "p1": ["memory_agent", "info_query_agent"],
        "p2": ["planning_agent"],
        "p3": ["execution_agent"],
        "summarize": ["summarization_agent"]
    },
    "info_query": {
        "p1": ["info_query_agent"],
        "p2": [],
        "p3": [],
        "summarize": ["summarization_agent"]
    },
    "preference_manage": {
        "p1": ["memory_agent", "preference_agent"],
        "p2": [],
        "p3": [],
        "summarize": ["summarization_agent"]
    },
    "memory_query": {
        "p1": ["memory_agent"],
        "p2": [],
        "p3": [],
        "summarize": ["summarization_agent"]
    }
}
```

---

## 信息查询 (InfoQueryAgent)

**无模拟数据**: 只返回真实查询结果或错误信息。

| 查询类型 | 触发关键词 | 返回内容 |
|----------|-----------|----------|
| 天气 | "天气"、"weather"、"温度" | 温度、天气状况、湿度、风力 |
| 时间 | "时间"、"几点"、"now" | 当前时间 HH:MM:SS |
| 日期 | "日期"、"几号"、"今天" | 当前日期、工作日 |
| 交通 | "交通"、"路况"、"堵车" | 拥堵程度、道路状况 |
| 搜索 | "搜索xxx"、"查一下xxx" | 相关结果摘要 |

**Skill集成**: 通过 `generic_skill.py` 自动加载Skills目录下的工具：
- `weather-query` - 天气查询
- `baidu-map-webapi` - 地图/交通
- `star-hotel` - 酒店查询
- `copey-flight-tracker` - 航班追踪

---

## 记忆系统

### 短期记忆 (Redis)

```
对话状态（Session级）
├── PreferenceAgent查询结果缓存
└── Agent执行历史
```

### 长期记忆 (PostgreSQL)

```
用户偏好 (user_preferences)
├── category: 酒店/交通/餐饮/景点
├── preference_key: 品牌/方式/菜系
├── preference_value: 具体值
├── confidence: 置信度 (0.0-1.0)
├── source: conversation/travel_history/explicit
└── metadata: 附加信息
```

### MemoryAgent 智能记忆

使用LLM分析对话内容：

1. **判断是否存储**: 用户是否在表达偏好/满意建议
2. **提取偏好信息**: 结构化提取（酒店、交通、餐饮等）
3. **更新策略**: 追加新偏好 vs 覆盖旧偏好
4. **自动存储**: TravelHistory、UserPreference等

**注意**: 对于简单查询（如查天气），MemoryAgent不会激活存储。

---

## 外部操作执行 (ExecutionAgent)

**无模拟数据**: 只返回真实执行结果或错误信息。

| 操作类型 | 触发关键词 | 说明 |
|----------|-----------|------|
| 订票 | "订机票"、"订酒店" | 需接入真实API |
| 设置闹钟 | "定闹钟"、"设闹钟" | 可调用系统API |
| 创建提醒 | "日程提醒"、"meeting" | 需接入日历API |
| 发送通知 | "发通知"、"发邮件" | 需接入通知服务 |

当前版本机票/酒店API未接入时，返回明确错误而非模拟成功。

---

## Skill系统

### GenericSkillLoader

自动扫描 `skills/` 目录，加载标准格式的Skill（无需agent.py）：

```
skills/
├── qweather-1.0.0/          # 和风天气
│   ├── SKILL.md              # Skill定义（必需）
│   └── scripts/              # 工具脚本
│       ├── weather_now.py
│       └── weather_forecast.py
├── baidu-map-webapi-1.0.7/   # 百度地图
├── tool_skill/               # 通用工具
└── notification_skill/       # Windows通知
```

### SKILL.md 格式

```markdown
---
name: weather-query
version: 1.0.0
description: 天气查询工具
tools: ["weather_now", "weather_forecast"]
functions: ["weather_now", "weather_forecast"]
trigger_keywords: ["天气", "weather", "温度"]
---

# 天气查询Skill

## 功能说明
查询指定城市的当前天气或预报...

## 工具列表
- weather_now: 当前天气
- weather_forecast: 天气预报
```

### 调用流程

```
InfoQueryAgent._query_weather()
    ↓
GenericSkillLoader.get_skill("weather-query")
    ↓
GenericSkillLoader.invoke_tool(tool, {"location": "北京"})
    ↓
subprocess.run(["python", "weather_now.py", "--location", "北京"])
    ↓
返回JSON结果
```

---

## LLM 配置

### 方式1: MiniMax API (当前使用)

```python
LLM_BACKEND = "minimax"
MINIMAX_CONFIG = {
    "api_key": os.getenv("MINIMAX_API_KEY", ""),
    "model_name": "MiniMax-M2.7",
    "base_url": "https://api.minimax.chat/v1",
}
```

### 方式2: 阿里云百炼 (Qwen3.5)

```python
LLM_BACKEND = "dashscope"
DASHSCOPE_CONFIG = {
    "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
    "model_name": "qwen3-14b",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
```

### 方式3: OpenAI兼容API (vLLM本地部署)

```python
LLM_BACKEND = "openai"
OPENAI_COMPAT_CONFIG = {
    "api_key": "not-required",
    "model_name": "your-trained-model",
    "base_url": "http://localhost:8000/v1",
}
```

环境变量方式：

```bash
export LLM_BACKEND=minimax
export MINIMAX_API_KEY=your-api-key
```

---

## 数据库配置

### PostgreSQL (长期记忆)

```bash
export PG_HOST=localhost
export PG_PORT=5432
export PG_DATABASE=multi_agent
export PG_USER=postgres
export PG_PASSWORD=postgres
```

### Redis (短期记忆)

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

**注意**: 连接失败时自动使用文件存储作为后备。

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
export LLM_BACKEND=minimax
export MINIMAX_API_KEY=your-api-key

# 启动服务
python main.py

# 访问 http://localhost:8000
```

---

## 目录结构

```
assistant/
├── agents/                       # 核心Agent (8个)
│   ├── intention_agent.py         # 意图识别
│   ├── orchestration_agent.py     # 调度（优先级+并行）
│   ├── memory_agent.py             # 智能记忆
│   ├── preference_agent.py         # 偏好管理
│   ├── info_query_agent.py         # 信息查询
│   ├── execution_agent.py          # 外部执行
│   ├── planning_agent.py           # 日程规划
│   └── summarization_agent.py      # 对话总结
├── skills/                        # Skill插件
│   ├── generic_skill.py           # Skill加载器
│   ├── tool_skill/                # 通用工具
│   ├── notification_skill/        # Windows通知
│   ├── qweather-1.0.0/            # 天气查询
│   ├── baidu-map-webapi-1.0.7/    # 百度地图
│   └── ...                        # 更多Skills
├── memory/                        # 记忆系统
│   ├── short_term.py              # Redis短期记忆
│   └── long_term.py               # PostgreSQL长期记忆
├── api/routes.py                  # FastAPI路由
├── core/                          # 核心配置
│   ├── config.py                  # 配置文件
│   └── llm_client.py              # LLM客户端
├── docs/                         # 技术文档
│   ├── TROUBLESHOOTING.md         # 问题排查
│   └── NOTIFICATION_SKILL.md      # 通知Skill使用
├── frontend/index.html            # 对话界面
├── main.py                        # 入口
└── requirements.txt
```

---

## 技术文档

| 文档 | 内容 |
|------|------|
| `docs/TROUBLESHOOTING.md` | 错误处理、限流、会话管理、Badcase处理 |
| `docs/NOTIFICATION_SKILL.md` | Windows通知和Microsoft Todo使用指南 |