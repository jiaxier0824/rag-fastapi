from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 百炼 API Key 仅从本地 .env 读取，不能提交到仓库。
    dashscope_api_key: str = ""

    # MySQL配置
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str
    db_name: str = "rag_project"

    # 原始文件存储配置
    upload_dir: str = "./uploads"

    # Chroma向量数据库配置
    chroma_collection_name: str = "rag"
    chroma_persist_dir: str = "./chroma_db"

    # 聊天历史文件配置
    chat_history_dir: str = "./chat_history"

    # 文本切分配置
    chunk_size: int = 1000
    chunk_overlap: int = 100
    separators: list[str] = Field(
        default_factory=lambda: [
            "\n\n",
            "\n",
            ".",
            "!",
            "?",
            "。",
            "！",
            "？",
            " ",
            ""
        ]
    )

    # 检索与模型配置
    retrieval_top_k: int = 3
    retrieval_vector_top_k: int = 5
    retrieval_keyword_top_k: int = 5
    retrieval_rrf_k: int = 60
    retrieval_candidate_top_k: int = 6
    rerank_enabled: bool = False
    rerank_model_name: str = "qwen3-rerank"
    rerank_top_n: int = 3
    rerank_timeout_seconds: float = 10.0
    dashscope_workspace_id: str | None = None
    dashscope_region: str = "cn-beijing"
    observability_log_path: str = "./logs/rag_requests.jsonl"
    embedding_model_name: str = "text-embedding-v4"
    # 课程资料问答优先响应速度；复杂推理留给上游 Agent 按需处理。
    chat_model_name: str = "qwen-turbo"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
