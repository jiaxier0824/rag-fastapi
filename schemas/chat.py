from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="用户问题"
    )

    session_id: str = Field(
        default="user_001",
        description="用户会话编号"
    )


class SourceReference(BaseModel):
    filename: str = Field(
        description="本次回答使用的知识库来源文件名"
    )


class ChatResponse(BaseModel):
    answer: str = Field(
        description="AI生成的回答"
    )
    session_id: str = Field(
        description="用户会话编号"
    )
    trace_id: str = Field(
        description="本次请求的唯一调用链编号，用于日志排查"
    )
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="本次回答使用的知识库来源"
    )
