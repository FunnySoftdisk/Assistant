# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

基于AgentScope框架的多Agent智能助手系统，实现意图识别、调度、偏好查询、信息查询、外部执行、日程规划、总结对话、记忆更新等功能。采用"优先级+并行"混合调度模式，FastAPI后端 + 对话式前端。

---

## Architecture

### 核心Agent (agents/)

| Agent | 职责 | 数据存储 |
|-------|------|----------|
| `intention_agent.py` | 语义理解，意图识别 | - |
| `orchestration_agent.py` | 调度中心，优先级+并行 | - |
| `preference_agent.py` | 偏好查询/管理 | Redis缓存 + PostgreSQL |
| `info_query_agent.py` | 天气、时间、交通查询 | - |
| `execution_agent.py` | 订票、定闹钟等操作 | - |
| `planning_agent.py` | 行程规划生成 | 整合偏好信息 |
| `summarization_agent.py` | 对话总结 | - |
| `memory_agent.py` | 记忆存储/更新 | **LLM驱动智能分析** |

### 记忆架构

```
短期记忆 (Redis)
├── 对话状态（Session级）
├── PreferenceAgent查询结果缓存
└── Agent执行历史

长期记忆 (PostgreSQL)
├── 用户偏好 (user_preferences)
├── 行程历史 (travel_history)
└── 对话摘要 (conversation_summary)
```

### 调度模式

```
用户消息 → IntentionAgent → OrchestrationAgent
                                    ↓
              ┌─ Priority 1 (并行) ──────────┐
              │  PreferenceAgent (查偏好)    │
              │  InfoQueryAgent (查天气等)    │
              │  ExecutionAgent (外部操作)     │
              └───────────────────────────────┘
                                    ↓ (等待完成)
              ┌─ Priority 2 (依赖P1) ────────┐
              │  PlanningAgent (生成行程)      │
              │  SummarizationAgent (总结)     │
              └───────────────────────────────┘
                                    ↓
                              MemoryAgent
                         (智能分析 + 存储记忆)
                                    ↓
                               返回结果
```

---

## MemoryAgent 智能记忆

MemoryAgent 使用LLM分析对话内容：

1. **判断是否存储**：用户是否在表达偏好/满意建议
2. **提取偏好信息**：结构化提取（酒店、交通、餐饮等）
3. **更新策略**：追加新偏好 vs 覆盖旧偏好
4. **自动存储**：TravelHistory、UserPreference等

---

## Agent职责详解

### InfoQueryAgent - 外部信息查询
- 天气查询（需接入天气API）
- 当前时间/日期
- 交通路况（需接入地图API）
- 联网搜索

### ExecutionAgent - 外部操作执行
- 订票（机票、酒店）- 需接入API
- 设置闹钟/提醒
- 发送通知

### PlanningAgent - 日程规划
- 根据目的地和时间生成行程
- 整合用户偏好（酒店偏好、交通偏好）
- 生成时间安排、预算估算、注意事项

### PreferenceAgent - 偏好管理
- 查询偏好（Redis缓存优先）
- 更新偏好（追加/覆盖）
- 与LongTermMemory联动

---

## LLM Configuration

### 方式1: 阿里云百炼 (Qwen3.5)
编辑 `core/config.py`:
```python
LLM_BACKEND = "dashscope"
DASHSCOPE_CONFIG = {
    "api_key": "your-api-key",
    "model_name": "qwen3-14b",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
```

### 方式2: 本地vLLM部署
```python
LLM_BACKEND = "openai"
OPENAI_COMPAT_CONFIG = {
    "model_name": "your-trained-model",
    "base_url": "http://localhost:8000/v1",
}
```

---

## Commands

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py

# 访问 http://localhost:8000
```

---

## Skill预留位置

已在各Agent中预留TODO标记：

| Agent | 预留Skill |
|-------|-----------|
| MemoryAgent | preference_learning_skill, memory_consolidation_skill |
| PreferenceAgent | preference_learning_skill, preference_suggestion_skill |
| InfoQueryAgent | weather_api_skill, map_api_skill, news_skill, stock_skill |
| ExecutionAgent | flight_booking_skill, hotel_booking_skill, alarm_system_skill, calendar_api_skill |
| PlanningAgent | budget_planning_skill, route_optimization_skill, packing_skill |

---

## 目录结构

```
assistant/
├── agents/                       # 核心Agent (8个)
│   ├── intention_agent.py         # 意图识别
│   ├── orchestration_agent.py     # 调度
│   ├── preference_agent.py        # 偏好管理
│   ├── info_query_agent.py        # 信息查询
│   ├── execution_agent.py         # 外部执行
│   ├── planning_agent.py          # 日程规划
│   ├── summarization_agent.py     # 对话总结
│   └── memory_agent.py            # 智能记忆
├── skills/                        # Skill插件
├── memory/                        # 记忆系统
│   ├── short_term.py              # Redis短期记忆
│   └── long_term.py               # PostgreSQL长期记忆
├── api/routes.py                  # FastAPI路由
├── models/schemas.py              # 数据模型
├── core/                          # 配置 + LLM客户端
├── frontend/index.html            # 对话界面
├── llm_local/                     # Qwen3-8B微调项目
├── main.py                        # 入口
└── requirements.txt
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

连接失败时会自动使用文件存储作为后备。