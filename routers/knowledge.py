"""知识文件相关 HTTP 接口。

Router 的边界是 HTTP：接收 multipart 文件、校验参数、把 Service 结果转换为响应 Schema。
它不直接操作 MySQL、文件系统或 Chroma，避免接口层长成难维护的业务脚本。
"""

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile
)
from sqlalchemy.ext.asyncio import AsyncSession

from config.database import get_db
from dependencies import get_knowledge_file_service
from schemas.knowledge import (
    KnowledgeDeleteResponse,
    KnowledgeFileListResponse,
    KnowledgeUploadResponse
)
from services.knowledge_file import (
    KnowledgeFileService
)


router = APIRouter(
    prefix="/api/knowledge",
    tags=["知识文件管理"]
)


@router.post(
    "/upload",
    response_model=KnowledgeUploadResponse
)
async def upload_knowledge(
    file: UploadFile = File(...),
    service: KnowledgeFileService = Depends(
        get_knowledge_file_service
    ),
    db: AsyncSession = Depends(get_db)
) -> KnowledgeUploadResponse:
    """接收 txt/md 文件，校验后交给 KnowledgeFileService 完成上传。"""
    # Router 只做 HTTP 输入/输出处理；上传、查重、三处存储由 Service 负责。
    filename = file.filename or "unknown.txt"

    # 当前解析链只支持 UTF-8 文本；PDF、Word 等格式需要先增加专门解析器。
    if not filename.lower().endswith((
        ".txt",
        ".md"
    )):
        raise HTTPException(
            status_code=400,
            detail="目前只支持txt和md文件"
        )

    # bytes 用于 MD5 和保存原始文件；解码后的 str 用于切分与向量化。
    content_bytes = await file.read()

    if not content_bytes:
        raise HTTPException(
            status_code=400,
            detail="上传的文件不能为空"
        )

    try:
        content_text = content_bytes.decode(
            "utf-8"
        )
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="文件必须使用UTF-8编码"
        ) from error

    # Depends 已注入缓存的 Service，以及当前请求专属的数据库 Session。
    file_record, created = await service.upload_file(
        db=db,
        filename=filename,
        content_type=(
            file.content_type or "text/plain"
        ),
        content_bytes=content_bytes,
        content_text=content_text
    )

    if created:
        message = "文件上传并写入知识库成功"
    else:
        message = "相同内容的文件已经存在，本次跳过"

    return KnowledgeUploadResponse(
        filename=file_record.filename,
        message=message
    )


@router.get(
    "/files",
    response_model=KnowledgeFileListResponse
)
async def get_knowledge_files(
    page: int = Query(
        default=1,
        ge=1,
        description="页码"
    ),
    page_size: int = Query(
        default=20,
        ge=1,
        le=100,
        description="每页数量"
    ),
    service: KnowledgeFileService = Depends(
        get_knowledge_file_service
    ),
    db: AsyncSession = Depends(get_db)
) -> KnowledgeFileListResponse:
    """分页读取已管理的知识文件，不读取 Chroma 中的具体切片。"""
    # page 是页面编号；offset 才是 SQL 中应跳过的“记录条数”。
    offset = (page - 1) * page_size

    file_records, total = (
        await service.list_files(
            db=db,
            offset=offset,
            limit=page_size
        )
    )

    return KnowledgeFileListResponse(
        items=file_records,
        total=total
    )


@router.delete(
    "/files/{file_id}",
    response_model=KnowledgeDeleteResponse
)
async def delete_knowledge_file(
    file_id: str,
    service: KnowledgeFileService = Depends(
        get_knowledge_file_service
    ),
    db: AsyncSession = Depends(get_db)
) -> KnowledgeDeleteResponse:
    """按 file_id 删除原始文件、MySQL 管理记录与 Chroma 切片。"""
    # file_id 是三个存储位置的关联键：MySQL 记录、本地文件名、Chroma metadata。
    file_record, deleted_chunks = (
        await service.delete_file(
            db=db,
            file_id=file_id
        )
    )

    if file_record is None:
        raise HTTPException(
            status_code=404,
            detail="知识文件不存在"
        )

    return KnowledgeDeleteResponse(
        file_id=file_record.file_id,
        filename=file_record.filename,
        deleted_chunks=deleted_chunks,
        message="知识文件删除成功"
    )
