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

class ChatResponse(BaseModel):
    answer: str = Field(
        description="AI生成的回答"
    )
    session_id: str = Field(
        description="用户会话编号"
    )