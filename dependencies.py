"""FastAPI 的依赖注入中心。

这里是整个项目创建“长期复用对象”的唯一入口。依赖关系是：

Embedding -> VectorStoreService -> KnowledgeBaseService -> KnowledgeFileService
                              -> RagService

Router 只声明 ``Depends(get_xxx_service)``，不负责知道对象怎样创建。这样可以避免
每个请求反复创建 Embedding、Chroma 连接和 RAG 链，也能保证上传与问答使用同一份
向量库。
"""

from functools import lru_cache

from langchain_community.embeddings import DashScopeEmbeddings

from config.settings import settings
from knowledge_base import KnowledgeBaseService
from rag import RagService
from services.knowledge_file import KnowledgeFileService
from stores.vector_store import VectorStoreService


@lru_cache
def get_embedding_model() -> DashScopeEmbeddings:
    """创建 Embedding 模型。

    它把文本转换为语义向量：上传时处理知识切片，问答时处理用户问题。
    ``@lru_cache`` 表示在当前 Python 进程生命周期内只创建一次；服务重启后会重新创建。
    """
    return DashScopeEmbeddings(
        model=settings.embedding_model_name
    )


@lru_cache
def get_vector_store_service() -> VectorStoreService:
    """提供唯一的 Chroma 操作服务，并复用上面的 Embedding 对象。

    此处把 ``get_embedding_model()`` 返回的对象作为构造参数传入。VectorStoreService
    因此“持有”Embedding 能力，但它并不拥有、也不会重复创建一个模型。
    """
    return VectorStoreService(
        embedding=get_embedding_model()
    )


@lru_cache
def get_knowledge_base_service() -> KnowledgeBaseService:
    """提供知识库服务：负责切分文本、构造 metadata，并委托向量库写入。"""
    return KnowledgeBaseService(
        vector_store_service=get_vector_store_service()
    )


@lru_cache
def get_knowledge_file_service() -> KnowledgeFileService:
    """提供文件服务：负责协调原始文件、MySQL 文件记录和 Chroma 切片。"""
    return KnowledgeFileService(
        knowledge_base_service=get_knowledge_base_service()
    )


@lru_cache
def get_rag_service() -> RagService:
    """提供 RAG 服务。

    注意它直接依赖 VectorStoreService，而不是 KnowledgeFileService：问答只需要“检索”，
    不需要文件上传、MD5 去重或 MySQL 写入能力。
    """
    return RagService(
        vector_store_service=get_vector_store_service()
    )
