"""知识文本的切分与 Chroma 写入协调层。"""

from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from stores.vector_store import VectorStoreService


class KnowledgeBaseService:
    """文本切分与向量知识库写入的协调层。

    它持有 VectorStoreService：自己负责“怎样切、附什么 metadata”，
    真正的 Chroma 写入和删除仍由 VectorStoreService 完成。
    """

    def __init__(
        self,
        vector_store_service: VectorStoreService
    ):
        """准备两种能力：文本切分器与已注入的向量库服务。"""

        # 保存同一个已注入的向量库服务对象，而不是在这里新建 Chroma。
        self.vector_store_service = vector_store_service

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=settings.separators,
            length_function=len
        )


    def split_text(self, data: str) -> list[str]:
        """按配置把完整文本拆成有重叠的 chunks，减少上下文在边界处断裂。"""
        return self.splitter.split_text(data)


    def add_texts(
        self,
        data: str,
        filename: str,
        file_id: str | None = None
    ) -> list[str]:
        """把一个完整文件的文本变为可检索的向量切片。

        此方法不保存原始文件，也不写 MySQL；它只处理“面向检索的知识”。返回值中的
        document_id 数量就是该文件的 chunk 数量，Service 会将其写回 MySQL 的 chunk_count。
        """
        # 一个文件只有一个 file_id；它的每个 chunk 都带上该 file_id，
        # 因而删除文件时可以从 Chroma 精确删除对应的全部 chunks。
        knowledge_chunks = self.split_text(data)

        # metadata 会跟随每一个 chunk 保存。source 用于回答时标识来源，file_id 用于删除。
        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "operator": "jiaxier"
        }

        if file_id is not None:
            metadata["file_id"] = file_id

        # 向量化由 Chroma 根据 VectorStoreService 中的 embedding 自动完成。
        document_ids = self.vector_store_service.add_texts(
            texts=knowledge_chunks,
            # 顺序号让后续检索能恢复相邻切片，PDF 等非 Markdown 原件也适用。
            metadatas=[
                {**metadata, "chunk_index": index}
                for index, _ in enumerate(knowledge_chunks)
            ]
        )

        return document_ids


    def delete_by_file_id(
        self,
        file_id: str
    ) -> int:
        return (
            self.vector_store_service.delete_by_file_id(
                file_id
            )
        )
