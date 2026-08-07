from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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
    retrieval_top_k: int = 1
    embedding_model_name: str = "text-embedding-v4"
    chat_model_name: str = "qwen3-max"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
