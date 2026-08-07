"""知识文件的业务服务层。

一份上传文件存在于三个位置：
1. uploads/：完整原始文件，便于留档和未来重新处理；
2. MySQL：文件管理记录（名称、MD5、状态、路径、chunk 数等）；
3. Chroma：用于语义检索的文本切片与向量。

由于这三处没有共同事务，本层负责按顺序协调，并在失败时尽力补偿已完成的操作。
"""

import asyncio
import hashlib

from pathlib import Path
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from crud import knowledge_file as knowledge_file_crud
from knowledge_base import KnowledgeBaseService
from models.knowledge_file import KnowledgeFile


class KnowledgeFileService:
    """编排一份知识文件在本地文件、MySQL、Chroma 三处的生命周期。"""

    def __init__(
        self,
        knowledge_base_service: KnowledgeBaseService
    ):
        self.knowledge_base_service = (
            knowledge_base_service
        )

        self.upload_dir = Path(
            settings.upload_dir
        )

        self.upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    async def upload_file(

        self,
        db: AsyncSession,
        filename: str,
        content_type: str,
        content_bytes: bytes,
        content_text: str
    ) -> tuple[KnowledgeFile, bool]:
        """上传一份知识文件。

        返回 ``(file_record, created)``：created=True 表示这次真的新建并入库；False 表示
        检测到相同 MD5 的旧文件，直接返回旧记录。内容 bytes 和 text 同时传入，是因为
        前者适合 MD5/原样写文件，后者适合切分和向量化。
        """

        # 先设为 None：若中途出错，except 可以判断哪些资源已经创建、需要清理。
        file_id: str | None = None
        storage_path: Path | None = None

        try:
            # 文件名可以重复，内容 MD5 才用来判断是否上传过同一份内容。
            md5_value = hashlib.md5(
                content_bytes
            ).hexdigest()

            existing_file = (
                await knowledge_file_crud.get_file_by_md5(
                    db=db,
                    md5=md5_value
                )
            )

            if existing_file is not None:
                # False 表示“重复跳过”，不是上传失败。
                return existing_file, False

            # file_id 是跨三处存储的业务关联键；它不同于 Chroma 为每个 chunk 分配的 document_id。
            file_id = str(uuid4())

            file_suffix = (
                Path(filename).suffix.lower()
            )

            storage_path = self.upload_dir / (
                f"{file_id}{file_suffix}"
            )

            # 本地文件写入是阻塞 I/O；放到线程中避免阻塞 FastAPI 事件循环。
            await asyncio.to_thread(
                storage_path.write_bytes,
                content_bytes
            )

            # 先插入 processing 状态的 MySQL 记录。flush 后获得完整 ORM 对象，但尚未 commit。
            file_record = (
                await knowledge_file_crud.create_file(
                    db=db,
                    file_id=file_id,
                    filename=filename,
                    content_type=content_type,
                    storage_path=str(
                        storage_path.resolve()
                    ),
                    md5=md5_value,
                    file_size=len(content_bytes)
                )
            )

            # 切分、Embedding 和 Chroma 写入同样可能阻塞，因此在线程中执行。
            document_ids = await asyncio.to_thread(
                self.knowledge_base_service.add_texts,
                data=content_text,
                filename=filename,
                file_id=file_id
            )

            # Chroma 成功后才把文件状态改为 completed，并记录实际被切出的 chunk 数量。
            file_record = (
                await knowledge_file_crud.update_file_status(
                    db=db,
                    file_record=file_record,
                    status="completed",
                    chunk_count=len(document_ids)
                )
            )

            # 到三处写入都成功后，才最终提交 MySQL 事务。
            await db.commit()

            return file_record, True

        except Exception:
            # rollback 只回滚 MySQL；Chroma 和本地文件需要在下面手动补偿清理。
            await db.rollback()

            if file_id is not None:
                try:
                    await asyncio.to_thread(
                        self.knowledge_base_service.delete_by_file_id,
                        file_id
                    )
                except Exception:
                    pass

            if storage_path is not None:
                try:
                    await asyncio.to_thread(
                        storage_path.unlink,
                        missing_ok=True
                    )
                except Exception:
                    pass

            raise

    async def delete_file(
        self,
        db: AsyncSession,
        file_id: str
    ) -> tuple[KnowledgeFile | None, int]:
        """删除一份文件在 Chroma、本地目录和 MySQL 的全部痕迹。

        找不到 file_id 时返回 ``(None, 0)``，由 Router 转换为 HTTP 404。删除失败不会悄悄
        吞掉：会尝试把 MySQL 状态记录成 ``delete_failed``，然后重新抛出异常。
        """

        # 先把状态提交为 deleting：若删除过程意外中断，数据库仍留下可排查的状态。
        file_record = (
            await knowledge_file_crud.get_file_by_file_id(
                db=db,
                file_id=file_id
            )
        )

        if file_record is None:
            return None, 0

        file_record = (
            await knowledge_file_crud.update_file_status(
                db=db,
                file_record=file_record,
                status="deleting"
            )
        )

        # 这次 commit 让 deleting 状态可见；即使进程中断，也能知道文件正处于何种阶段。
        await db.commit()

        try:
            # 删除顺序：Chroma 切片 -> 原始本地文件 -> MySQL 文件记录。
            deleted_chunks = await asyncio.to_thread(
                self.knowledge_base_service.delete_by_file_id,
                file_id
            )

            storage_path = Path(
                file_record.storage_path
            )

            await asyncio.to_thread(
                storage_path.unlink,
                missing_ok=True
            )

            await knowledge_file_crud.delete_file(
                db=db,
                file_record=file_record
            )

            await db.commit()

            return file_record, deleted_chunks

        except Exception as error:
            # 若失败，记录 delete_failed，方便之后人工或程序重试。
            await db.rollback()

            failed_record = (
                await knowledge_file_crud.get_file_by_file_id(
                    db=db,
                    file_id=file_id
                )
            )

            if failed_record is not None:
                await knowledge_file_crud.update_file_status(
                    db=db,
                    file_record=failed_record,
                    status="delete_failed",
                    error_message=str(error)
                )

                await db.commit()

            raise

    async def list_files(
        self,
        db: AsyncSession,
        offset: int = 0,
        limit: int = 20
    ) -> tuple[list[KnowledgeFile], int]:
        """读取文件列表。Service 保持统一入口，实际 SQL 查询委托 CRUD 层。"""
        return await knowledge_file_crud.list_files(
            db=db,
            offset=offset,
            limit=limit
        )
