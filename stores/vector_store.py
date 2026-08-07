"""Chroma 向量数据库的访问层。

上层只需要理解三个动作：写入文本、按 file_id 删除、拿到 Retriever 检索器。
Embedding、Chroma 的持久化目录和 collection 名称等基础设施细节集中在本文件。
"""

from pathlib import Path

from langchain_chroma import Chroma

from config.settings import settings


class VectorStoreService:
    """集中管理唯一的 Chroma 操作对象。

    Chroma 内部保存每个文本切片、它的向量、metadata 和 document_id；
    本类不关心文件上传或 MySQL，只负责向量库读写。
    """

    def __init__(self, embedding):
        """初始化 Chroma 客户端。

        参数 ``embedding`` 是 Dependencies 注入的 DashScopeEmbeddings 实例。Chroma 在
        ``add_texts`` 时用它把每个文本切片转为向量；在 Retriever 搜索时也用它把问题
        转为向量。因此写入和检索使用的是同一种语义坐标体系。
        """
        # 这里保存的是注入进来的 Embedding 对象，不会重新创建模型。
        self.embedding = embedding

        Path(settings.chroma_persist_dir).mkdir(
            parents=True,
            exist_ok=True
        )

        # Chroma 会使用 embedding_function 将 texts 自动向量化，并持久化到目录。
        self.vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=self.embedding,
            persist_directory=settings.chroma_persist_dir
        )

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict]
    ) -> list[str]:
        """将一批文本切片写入 Chroma，并返回每个切片对应的 document_id。

        输入的 ``texts`` 是切分后的原文，``metadatas`` 是与其一一对应的附加信息。
        调用 Chroma 后，真正保存的是：文本切片、由 Embedding 得到的向量、metadata 和
        document_id。调用者只拿 document_id 回执，数据本身已经持久化在 chroma_db 中。
        """
        return self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas
        )

    def delete_by_file_id(self, file_id: str) -> int:
        """删除属于一个原始文件的全部向量切片，并返回删除数量。"""
        # 一个原始文件会被切成多个文档；通过共同的 file_id 找齐再批量删除。
        stored_data = self.vector_store.get(
            where={
                "file_id": file_id
            },
            include=[]
        )

        document_ids = stored_data.get("ids", [])

        if document_ids:
            self.vector_store.delete(ids=document_ids)

        return len(document_ids)

    def get_retriever(self):
        """将 Chroma 适配为 LangChain Retriever，而不是立刻发起检索。

        真正搜索发生在 RAG 链运行、Retriever 收到用户问题时；此处仅配置搜索规则。
        """
        # Retriever 是 LangChain 的标准检索接口，便于直接接入 RAG 链。
        # k 表示每次问答取最相似的前 k 个文本切片。
        return self.vector_store.as_retriever(
            search_kwargs={
                "k": settings.retrieval_top_k
            }
        )
