# Multi-Agent Assistant 技术文档

## 概述

本文档记录了 Multi-Agent 智能助手系统从 Demo 到生产级别所解决的技术问题和方案。

---

## 1. 错误处理与优雅降级

### 问题背景
- LLM 调用可能超时失败
- 单个 Agent 异常不应影响整体流程
- 用户应收到友好的错误提示而非技术异常

### 解决方案

#### 1.1 LLM 客户端超时与重试

```python
# core/llm_client.py
@async_retry(max_attempts=3, base_delay=1.0, backoff=2.0)
async def chat(self, messages, ...):
    """带指数退避重试的 LLM 调用"""
    try:
        response = await self.client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException:
        raise LLMTimeoutError(f"Request timeout after {self.timeout}s")
```

**重试策略**：
- 最多 3 次重试
- 指数退避：1s → 2s → 4s
- 超时时间：默认 60 秒

#### 1.2 Agent 超时保护

```python
# 所有 Agent 的 reply 方法
try:
    result = await asyncio.wait_for(
        self._execute_task(task),
        timeout=30.0  # 30秒超时
    )
except asyncio.TimeoutError:
    return {"error": "执行超时，请稍后重试"}
```

#### 1.3 关键词回退机制

意图识别 Agent 在 LLM 失败时自动降级到关键词匹配：

```python
def _classify_by_keywords(self, query: str) -> Dict:
    """关键词回退意图分类"""
    intent_mapping = {
        "travel_planning": ["规划", "行程", "去", "旅游"],
        "info_query": ["天气", "搜索", "查询"],
        # ...
    }
    # 计算关键词得分，返回最高匹配意图
```

#### 1.4 调度层容错

```python
# orchestration_agent.py
async def _execute_parallel(self, tasks, timeout=30.0):
    """单个 Agent 失败不影响其他 Agent"""
    results = await asyncio.gather(
        *[execute_task_safe(agent, input) for agent, input in tasks],
        return_exceptions=True  # 关键：异常不抛出
    )
    # 处理结果时检查是否为异常
```

---

## 2. Rate Limiting 与请求验证

### 问题背景
- API 可能被恶意频繁调用
- 过长的输入可能导致内存问题
- 需要防止注入攻击

### 解决方案

#### 2.1 频率限制

```python
# api/routes.py
class RateLimiter:
    def is_allowed(self, key, max_per_minute=60, max_per_day=1000):
        # 每分钟检查
        recent = [t for t in self._requests[key] if now - t < 60]
        if len(recent) >= max_per_minute:
            return False, "请求过于频繁"
        # 每日检查
        if self._daily_counts.get(key, 0) >= max_per_day:
            return False, "今日请求次数已达上限"
```

**限制策略**：
| 维度 | 限制 |
|------|------|
| 每分钟 | 60 次 |
| 每日 | 1000 次 |

#### 2.2 输入验证

```python
# core/utils.py
def validate_message(message: str, max_length: int = 2000):
    """验证消息格式"""
    if not message:
        return False, "消息不能为空"
    if len(message) > max_length:
        return False, f"消息长度不能超过{max_length}字符"

    # 检查危险字符
    dangerous_patterns = [r'<script', r'javascript:', r'onerror=']
    for pattern in dangerous_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return False, "消息包含非法字符"
    return True, ""
```

---

## 3. 会话管理

### 问题背景
- 会话数据无限增长导致内存溢出
- 需要会话过期机制
- 多轮对话上下文管理

### 解决方案

#### 3.1 会话状态结构

```python
@dataclass
class ConversationState:
    session_id: str
    messages: List[Dict]  # 限制最近 20 轮
    current_intent: str
    entities: Dict[str, Any]  # 关键实体
    preferences_cache: Dict[str, Any]
    created_at: float
    last_active: float
```

#### 3.2 TTL 自动过期

```python
# Redis
redis_client.setex(key, ttl=3600, value=data)  # 1小时过期

# 本地后备
def cleanup_expired(self):
    """清理过期数据"""
    current_time = time.time()
    expired_keys = [
        key for key, data in self._local_store.items()
        if data.get("expires_at", 0) < current_time
    ]
```

#### 3.3 消息历史截断

```python
def save_conversation_state(self, session_id, state):
    # 只保留最近 N 条消息
    state.messages = state.messages[-self.max_history:]
```

---

## 4. 数据库连接池

### 问题背景
- 每次请求创建新连接开销大
- 数据库连接数有限
- 断线需要自动重连

### 解决方案

#### 4.1 PostgreSQL 连接池

```python
# memory/long_term.py
class ConnectionPool:
    def __init__(self, config, min_conn=2, max_conn=10):
        self._pool = pool.ThreadedConnectionPool(
            min_conn, max_conn,
            host=config["host"],
            port=config["port"],
            database=config["database"]
        )

    @contextmanager
    def connection(self):
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)  # 归还连接
```

#### 4.2 Redis 连接池

```python
# memory/short_term.py
class RedisPool:
    def __init__(self, redis_config, max_connections=50):
        self._pool = ConnectionPool(
            host=redis_config["host"],
            port=redis_config["port"],
            max_connections=max_connections
        )
```

#### 4.3 文件存储后备

当数据库不可用时，自动降级到文件存储：

```python
def save_preference(self, preference):
    conn = self._get_connection()
    if conn is None:
        return self._save_preference_file(preference)  # 文件后备
    # 正常保存到数据库
```

---

## 5. 前端健壮性

### 问题背景
- 网络抖动导致请求失败
- 用户不知道发生了什么
- 输入过长未被提示

### 解决方案

#### 5.1 自动重试机制

```javascript
// 前端 index.html
async function sendMessage() {
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            const response = await fetch('/api/chat', {...});
            // 成功处理
            return;
        } catch (error) {
            if (attempt < maxRetries) {
                await new Promise(r => setTimeout(r, 1000 * attempt));
            }
        }
    }
    // 所有重试失败，显示错误
}
```

#### 5.2 加载状态指示

```javascript
function showLoading() {
    // 显示加载动画
    chatBox.insertAdjacentHTML('beforeend', `
        <div class="loading-indicator">
            <div class="loading-spinner"></div>
            <span>正在处理...</span>
        </div>
    `);
}
```

#### 5.3 输入长度实时显示

```javascript
userInput.addEventListener('input', function() {
    const len = this.value.length;
    charCount.textContent = `${len}/2000`;
    if (len > 2000) charCount.classList.add('error');
});
```

#### 5.4 心跳检测

```javascript
setInterval(async () => {
    try {
        const response = await fetch('/api/health');
        if (response.ok) updateSystemStatus('online');
    } catch {
        updateSystemStatus('error', '网络异常');
    }
}, 30000);
```

---

## 6. Badcase 处理汇总

| 问题场景 | 原因 | 解决方案 |
|---------|------|---------|
| LLM 调用超时 | 网络问题或模型响应慢 | 30秒超时 + 关键词回退 |
| Redis/PostgreSQL 断连 | 服务不可用 | 自动降级到文件存储 |
| 频繁请求 | 恶意或异常调用 | Rate Limiting |
| 输入过长 | 占用过多内存 | 2000字符限制 + 截断 |
| 单个 Agent 失败 | 业务异常 | return_exceptions=True 隔离 |
| 网络抖动 | 临时故障 | 前端自动重试(2次) |
| 消息解析失败 | LLM 返回格式异常 | safe_json_parse 多级回退 |
| 组件初始化失败 | 依赖缺失 | 延迟初始化 + 友好错误 |

---

## 7. 监控与日志

### 健康检查端点

```
GET /api/health
Response: {
    "status": "ok",
    "agents_initialized": true,
    "timestamp": 1234567890
}
```

### 限流状态查询

```
GET /api/rate-limit-status?user_id=xxx
Response: {
    "remaining_minute": 55,
    "remaining_day": 980,
    "limit_per_minute": 60
}
```

---

## 8. 安全考虑

### 8.1 输入安全
- HTML/JS 标签过滤
- SQL 注入防护（使用参数化查询）
- XSS 防护（输出转义）

### 8.2 速率安全
- IP + UserID 双重维度限流
- 每日配额控制

### 8.3 敏感信息
- API Key 不硬编码，通过环境变量注入
- 日志脱敏处理

---

## 9. 性能优化建议

### 9.1 LLM 调用优化
- 使用连接池复用 HTTP 连接
- 批量请求合并（可选）
- 结果缓存（适合重复查询）

### 9.2 异步优化
- Agent 并行执行（asyncio.gather）
- 非关键路径不阻塞（记忆更新异步）

### 9.3 缓存策略
| 数据类型 | 缓存时间 | 说明 |
|---------|---------|------|
| 偏好查询结果 | 1小时 | 频繁访问 |
| 对话状态 | 1小时 | Session 级 |
| 搜索结果 | 5分钟 | 避免重复搜索 |

---

## 10. 部署建议

### 10.1 环境变量配置
```bash
# LLM 配置
export DASHSCOPE_API_KEY="your-key"
export MODEL_NAME="qwen3-8b"
export LLM_BACKEND="dashscope"

# 数据库配置
export PG_HOST="localhost"
export PG_PORT="5432"
export PG_DATABASE="multi_agent"
export PG_USER="postgres"
export PG_PASSWORD="password"

# Redis 配置
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
```

### 10.2 启动命令
```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py

# 或使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 10.3 Docker 部署（可选）
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```
