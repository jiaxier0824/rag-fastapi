from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Integer,
    String,
    Text,
    func
)
from sqlalchemy.orm import Mapped, mapped_column

from config.database import Base


class KnowledgeFile(Base):
    __tablename__ = "knowledge_files"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    file_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        index=True,
        nullable=False
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    content_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    storage_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False
    )

    md5: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        index=True,
        nullable=False
    )

    file_size: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(20),
        default="processing",
        nullable=False
    )

    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )