"""FastAPI 应用入口。

本文件只负责把应用“组装起来”：管理应用启停、注册 Router、提供健康检查。
具体的上传、数据库、向量检索和大模型调用都不应该写在这里；它们分别属于
Router、Service、Store 等更专门的层。
"""

from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.database import async_engine
from routers.chat import router as chat_router
from routers.knowledge import router as knowledge_router
from routers.agent_proxy import router as agent_proxy_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用生命周期。

    FastAPI 启动时会进入此函数，执行到 ``yield`` 前的代码；应用持续运行期间
    停在 ``yield`` 处；服务停止时继续执行 ``yield`` 后的清理代码。
    """
    yield

    # Uvicorn 停止时主动释放 SQLAlchemy 的连接池，避免连接残留。
    await async_engine.dispose()


app = FastAPI(
    title="RAG知识库API",
    description="提供知识库上传和RAG问答服务",
    version="1.0.0",
    lifespan=lifespan
)

# Router 只定义一组接口；在入口统一注册后，这些接口才属于 app。
app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(agent_proxy_router)

# 轻量前端与 API 共用同一个 FastAPI 服务。/docs 仍保留给开发者调试接口。
frontend_dir = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir / "static"), name="static")


@app.get("/", include_in_schema=False)
def study_search_page() -> FileResponse:
    """返回给普通用户使用的知识库页面，而不是 Swagger 调试页。"""
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "RAG服务运行正常"
    }
