"""RAG 问答 HTTP 接口。"""

from fastapi import APIRouter, Depends, Header
from uuid import uuid4

from dependencies import get_rag_service
from schemas.chat import ChatRequest, ChatResponse, SourceReference
from rag import RagService


router = APIRouter(
    prefix="/api/rag",
    tags=["RAG问答"]
)


@router.post(
    "/chat",
    response_model=ChatResponse
)
def chat(
        request: ChatRequest,
        rag_service: RagService = Depends(get_rag_service),
        x_trace_id: str | None = Header(default=None),
) -> ChatResponse:
    """执行一次带历史记录的 RAG 问答，并以 Schema 形式返回答案。"""
    # 独立调用时由 RAG 创建；被 Agent 调用时沿用同一条调用链。
    trace_id = x_trace_id or str(uuid4())
    answer, source_filenames = rag_service.ask(
        question=request.question,
        session_id=request.session_id,
        trace_id=trace_id,
    )

    return ChatResponse(
        answer=answer,
        session_id=request.session_id,
        trace_id=trace_id,
        sources=[
            SourceReference(filename=filename)
            for filename in source_filenames
        ],
    )
