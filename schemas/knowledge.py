from pydantic import BaseModel, Field, ConfigDict

from datetime import datetime

class KnowledgeUploadResponse(BaseModel):
    filename: str = Field(
        description="上传的文件名"
    )

    message: str = Field(
        description="知识库处理结果"
    )

class KnowledgeFileResponse(BaseModel):
    file_id: str
    filename: str
    content_type: str
    file_size: int
    status: str
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )


class KnowledgeFileListResponse(BaseModel):
    items: list[KnowledgeFileResponse]
    total: int


class KnowledgeDeleteResponse(BaseModel):
    file_id: str
    filename: str
    deleted_chunks: int
    message: str