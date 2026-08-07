"""KnowledgeFile 表的 CRUD 数据访问层。

本层只描述如何用 SQLAlchemy ORM 读写 MySQL，不承担上传流程、Chroma 写入或 HTTP 返回。
Service 调用这些函数，把多个基础操作组合成完整业务动作。
"""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.knowledge_file import KnowledgeFile


async def get_file_by_md5(
    db: AsyncSession,
    md5: str
) -> KnowledgeFile | None:
    """按内容 MD5 查找文件；MD5 唯一，因此结果最多一行。"""
    # ORM 先构造 SELECT；await execute 时才由数据库驱动真正向 MySQL 发起查询。
    statement = select(KnowledgeFile).where(
        KnowledgeFile.md5 == md5
    )

    result = await db.execute(statement)

    return result.scalar_one_or_none()


async def get_file_by_file_id(
    db: AsyncSession,
    file_id: str
) -> KnowledgeFile | None:
    """按业务 file_id 查询一条文件管理记录。"""
    statement = select(KnowledgeFile).where(
        KnowledgeFile.file_id == file_id
    )

    result = await db.execute(statement)

    return result.scalar_one_or_none()


async def create_file(
    db: AsyncSession,
    file_id: str,
    filename: str,
    content_type: str,
    storage_path: str,
    md5: str,
    file_size: int
) -> KnowledgeFile:
    """创建 processing 状态的 ORM 记录，但不在本函数中 commit。"""
    # add 把 ORM 对象加入当前 Session；flush 会执行 INSERT，但仍可在 commit 前 rollback。
    file_record = KnowledgeFile(
        file_id=file_id,
        filename=filename,
        content_type=content_type,
        storage_path=storage_path,
        md5=md5,
        file_size=file_size,
        status="processing",
        chunk_count=0
    )

    db.add(file_record)

    await db.flush()
    # refresh 从数据库重新读取该行，例如数据库生成的 created_at。
    await db.refresh(file_record)

    return file_record


async def list_files(
    db: AsyncSession,
    offset: int = 0,
    limit: int = 20
) -> tuple[list[KnowledgeFile], int]:
    """返回 ``(当前页 ORM 记录列表, 全部记录总数)``。"""
    # COUNT(*) 与分页列表分开查询：前者给前端 total，后者给当前页 items。
    count_statement = select(
        func.count()
    ).select_from(KnowledgeFile)

    count_result = await db.execute(
        count_statement
    )

    total = count_result.scalar_one()

    list_statement = (
        select(KnowledgeFile)
        .order_by(KnowledgeFile.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    list_result = await db.execute(
        list_statement
    )

    file_records = list(
        list_result.scalars().all()
    )

    return file_records, total


async def update_file_status(
    db: AsyncSession,
    file_record: KnowledgeFile,
    status: str,
    chunk_count: int = 0,
    error_message: str | None = None
) -> KnowledgeFile:
    """更新状态、chunk 数和错误信息；提交时机交给上层 Service。"""
    # flush 后仍在当前事务中；最终是否提交由 Service 决定。
    file_record.status = status
    file_record.chunk_count = chunk_count
    file_record.error_message = error_message

    await db.flush()
    await db.refresh(file_record)

    return file_record


async def delete_file(
    db: AsyncSession,
    file_record: KnowledgeFile
) -> None:
    """把 ORM 记录标记为删除并 flush；最终 commit 仍由 Service 执行。"""
    await db.delete(file_record)
    await db.flush()
