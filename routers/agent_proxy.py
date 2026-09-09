"""Proxy Agent SSE through the RAG app while keeping services separate."""

import json
import os
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/agent", tags=["Agent gateway"])
AGENT_API_BASE_URL = os.getenv("AGENT_API_BASE_URL", "http://agent:8001").rstrip("/")


class AgentTaskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=128)


@router.get("/health")
async def agent_health() -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{AGENT_API_BASE_URL}/health")
            response.raise_for_status()
        return JSONResponse({"status": "ok", "message": "Agent服务运行正常"})
    except httpx.HTTPError:
        return JSONResponse({"status": "unavailable", "message": "Agent服务暂时不可用"}, status_code=503)


@router.post("/chat/stream")
async def stream_agent_task(request: AgentTaskRequest) -> StreamingResponse:
    async def forward_events() -> AsyncIterator[bytes]:
        try:
            timeout = httpx.Timeout(120.0, connect=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{AGENT_API_BASE_URL}/api/agent/chat/stream",
                    json=request.model_dump(),
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
        except httpx.HTTPError:
            payload = json.dumps({"type": "error", "content": "Agent服务暂时不可用，请稍后重试。"}, ensure_ascii=False)
            yield f"event: message\ndata: {payload}\n\n".encode()

    return StreamingResponse(forward_events(), media_type="text/event-stream")
