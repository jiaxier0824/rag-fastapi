"""Chroma 向量数据库的访问层。

上层只需要理解三个动作：写入文本、按 file_id 删除、拿到 Retriever 检索器。
Embedding、Chroma 的持久化目录和 collection 名称等基础设施细节集中在本文件。
"""

import re
from time import perf_counter
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

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

    def search(
        self,
        query: str,
    ) -> list[Document]:
        """合并向量检索与关键词检索的结果，返回最终上下文切片。"""
        fusion_top_k = (
            settings.retrieval_candidate_top_k
            if settings.rerank_enabled
            else settings.retrieval_top_k
        )
        documents, _ = self.search_with_trace(
            query=query,
            top_k=fusion_top_k,
        )
        return documents

    def search_with_trace(
        self,
        query: str,
        top_k: int,
    ) -> tuple[list[Document], dict[str, float]]:
        """执行混合检索，并返回切片与各检索阶段耗时（毫秒）。

        正常问答仍使用 ``search``。此方法给离线评测和后续可观测性使用，
        因而能比较 RRF 基线与重排版本的实际延迟。
        """
        total_started_at = perf_counter()

        vector_started_at = perf_counter()
        vector_documents = self.vector_store.similarity_search(
            query=query,
            k=settings.retrieval_vector_top_k,
        )
        vector_elapsed_ms = (perf_counter() - vector_started_at) * 1000

        keyword_started_at = perf_counter()
        keyword_documents = self._keyword_search(query)
        keyword_elapsed_ms = (perf_counter() - keyword_started_at) * 1000

        fusion_started_at = perf_counter()
        documents = self._reciprocal_rank_fusion(
            ranked_document_lists=[vector_documents, keyword_documents],
            top_k=top_k,
        )
        fusion_elapsed_ms = (perf_counter() - fusion_started_at) * 1000

        return documents, {
            "vector_search_ms": round(vector_elapsed_ms, 2),
            "keyword_search_ms": round(keyword_elapsed_ms, 2),
            "rrf_fusion_ms": round(fusion_elapsed_ms, 2),
            "hybrid_total_ms": round(
                (perf_counter() - total_started_at) * 1000,
                2,
            ),
        }

    def _keyword_search(self, query: str) -> list[Document]:
        """用 BM25 从当前 Chroma 全量切片中召回包含关键词的内容。"""
        stored_data = self.vector_store.get(
            include=["documents", "metadatas"],
        )
        documents = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(
                stored_data.get("documents", []),
                stored_data.get("metadatas", []),
            )
            if text
        ]

        if not documents:
            return []

        tokenized_documents = [
            self._tokenize(document.page_content)
            for document in documents
        ]
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        bm25 = BM25Okapi(tokenized_documents)
        scores = bm25.get_scores(query_tokens)
        ranked_indexes = sorted(
            range(len(documents)),
            key=lambda index: scores[index],
            reverse=True,
        )

        return [
            documents[index]
            for index in ranked_indexes[:settings.retrieval_keyword_top_k]
            if scores[index] > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """把英文单词、数字和中文字符切成 BM25 可计算的 token。"""
        return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())

    @staticmethod
    def _document_key(document: Document) -> tuple[str, str]:
        """为同一文本切片生成稳定键，供两路召回结果去重。"""
        return document.page_content, str(document.metadata.get("file_id", ""))

    def _reciprocal_rank_fusion(
        self,
        ranked_document_lists: list[list[Document]],
        top_k: int,
    ) -> list[Document]:
        """使用 RRF 合并多路召回结果，避免单一路径主导最终排序。"""
        scores: dict[tuple[str, str], float] = {}
        documents_by_key: dict[tuple[str, str], Document] = {}

        for documents in ranked_document_lists:
            for rank, document in enumerate(documents, start=1):
                key = self._document_key(document)
                documents_by_key[key] = document
                scores[key] = scores.get(key, 0.0) + 1 / (
                    settings.retrieval_rrf_k + rank
                )

        ranked_keys = sorted(
            scores,
            key=lambda key: scores[key],
            reverse=True,
        )

        return [
            documents_by_key[key]
            for key in ranked_keys[:top_k]
        ]

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
