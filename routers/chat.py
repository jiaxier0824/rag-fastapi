"""RAG 问答 HTTP 接口。"""

from fastapi import APIRouter, Depends

from dependencies import get_rag_service
from schemas.chat import ChatRequest, ChatResponse
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
) -> ChatResponse:
    """执行一次带历史记录的 RAG 问答，并以 Schema 形式返回答案。"""
    # config 不属于提示词业务数据；它专门把 session_id 交给聊天历史包装器。
    session_config = {
        "configurable": {
            "session_id": request.session_id
        }
    }

    # input 是用户问题。RAG 链会据此检索 Chroma、组装提示词并调用大模型。
    answer = rag_service.chain.invoke(
        {
            "input": request.question
        },
        config=session_config
    )

    return ChatResponse(
        answer=answer,
        session_id=request.session_id
    )
