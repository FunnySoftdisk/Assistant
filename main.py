"""
FastAPI主入口
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from api.routes import router, init_agents
import uvicorn
import os

# 创建FastAPI应用
app = FastAPI(
    title="Multi-Agent Assistant",
    description="基于AgentScope的多Agent智能助手",
    version="1.0.0"
)

# 注册路由
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    # 确保数据目录存在
    os.makedirs("data/memory", exist_ok=True)
    os.makedirs("skills", exist_ok=True)

    # 初始化Agent
    init_agents()
    print("\n" + "="*50)
    print("✓ Multi-Agent Assistant 已启动")
    print("  前端: http://localhost:8000")
    print("  API:  http://localhost:8000/api")
    print("  Docs: http://localhost:8000/docs")
    print("="*50 + "\n")


@app.get("/", response_class=HTMLResponse)
async def index():
    """首页 - 对话界面"""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return """
        <html><body>
        <h1>Multi-Agent Assistant</h1>
        <p>Frontend not found.</p>
        </body></html>
        """


@app.get("/status")
async def status():
    """系统状态"""
    from skills.skill_registry import get_skill_registry

    registry = get_skill_registry()
    skills = registry.list_skills()

    return JSONResponse({
        "system": "Multi-Agent Assistant",
        "version": "1.0.0",
        "framework": "AgentScope",
        "status": "running",
        "agents": {
            "intention_agent": "意图识别",
            "orchestration_agent": "任务调度",
            "preference_agent": "偏好管理",
            "info_query_agent": "信息查询",
            "execution_agent": "外部执行",
            "planning_agent": "日程规划",
            "summarization_agent": "对话总结",
            "memory_agent": "记忆更新"
        },
        "skills": {
            s.name: {"version": s.version, "description": s.description}
            for s in skills.values()
        }
    })


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║        Multi-Agent Assistant (AgentScope)          ║
╠══════════════════════════════════════════════════════╣
║  前端: http://localhost:8000                        ║
║  API:  http://localhost:8000/api                    ║
║  Docs: http://localhost:8000/docs                   ║
╚══════════════════════════════════════════════════════╝
    """)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)