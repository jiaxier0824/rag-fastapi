"""Chroma 向量数据库的访问层。

上层只需要理解三个动作：写入文本、按 file_id 删除、拿到 Retriever 检索器。
Embedding、Chroma 的持久化目录和 collection 名称等基础设施细节集中在本文件。
"""

import re
from time import perf_counter
from pathlib import Path
from uuid import UUID

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from config.settings import settings


# rank_bm25 不负责文本预处理；两端统一去除英文虚词，保留课程编号、数字和中文。
_ENGLISH_STOPWORDS = frozenset(
    "a an and are as at be been being by can did do does for from had has "
    "have how if in into is it its may of on or our shall should that the "
    "their them these this those to was were what when where which who whom "
    "why will with would you your".split()
)


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

    def expand_context_windows(self, candidates: list[Document]) -> list[Document]:
        """合并同文件里连续命中的切片，再补一个后继切片恢复边界上下文。

        窗口只包含同一原始文件的真实文字，数量不会超过原候选数。
        新索引使用 chunk_index；旧索引先与上传原文重新切分的结果核对顺序。
        无法核对时保持原候选，不猜测相邻关系。
        """
        if not candidates:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=settings.separators,
            length_function=len,
        )
        chunks_by_file: dict[str, list[str] | None] = {}
        groups: dict[str, list[tuple[int, int, Document]]] = {}
        windows: list[tuple[int, Document]] = []
        for rank, document in enumerate(candidates):
            file_id = document.metadata.get("file_id")
            if not isinstance(file_id, str):
                windows.append((rank, document))
                continue
            try:
                UUID(file_id)
            except ValueError:
                windows.append((rank, document))
                continue

            if file_id not in chunks_by_file:
                stored = self.vector_store.get(
                    where={"file_id": file_id}, include=["documents", "metadatas"]
                )
                texts = stored.get("documents") or []
                metadatas = stored.get("metadatas") or []
                indexes = [meta.get("chunk_index") for meta in metadatas]
                if len(texts) == len(indexes) and sorted(
                    index for index in indexes if isinstance(index, int)
                ) == list(range(len(texts))):
                    chunks_by_file[file_id] = [
                        text for _, text in sorted(zip(indexes, texts))
                    ]
                else:
                    path = Path(settings.upload_dir) / f"{file_id}.md"
                    if not path.is_file():
                        chunks_by_file[file_id] = None
                    else:
                        try:
                            original = splitter.split_text(path.read_text(encoding="utf-8"))
                        except (OSError, UnicodeError):
                            chunks_by_file[file_id] = None
                        else:
                            chunks_by_file[file_id] = texts if texts == original else None

            chunks = chunks_by_file[file_id]
            if not chunks or chunks.count(document.page_content) != 1:
                windows.append((rank, document))
                continue
            groups.setdefault(file_id, []).append(
                (chunks.index(document.page_content), rank, document)
            )

        for file_id, matches in groups.items():
            chunks = chunks_by_file[file_id]
            assert chunks is not None
            ordered = sorted(matches)
            runs: list[list[tuple[int, int, Document]]] = []
            for match in ordered:
                if runs and match[0] == runs[-1][-1][0] + 1:
                    runs[-1].append(match)
                else:
                    runs.append([match])
            for run in runs:
                first_index = run[0][0]
                last_index = min(run[-1][0] + 1, len(chunks) - 1)
                earliest = min(run, key=lambda match: match[1])
                windows.append((earliest[1], Document(
                    page_content="\n".join(chunks[first_index:last_index + 1]),
                    metadata=earliest[2].metadata,
                )))
        return [document for _, document in sorted(windows, key=lambda item: item[0])]

    def search_with_trace(
        self,
        query: str,
        top_k: int,
        source_hint: str | None = None,
    ) -> tuple[list[Document], dict[str, float]]:
        """执行混合检索，并返回切片与各检索阶段耗时（毫秒）。

        正常问答直接使用此方法；同时记录向量检索、BM25 和 RRF 的耗时。
        """
        total_started_at = perf_counter()

        scope_started_at = perf_counter()
        scoped_documents = self._explicit_source_search(query, source_hint, top_k)
        scope_elapsed_ms = (perf_counter() - scope_started_at) * 1000
        if scoped_documents:
            return scoped_documents, {
                "vector_search_ms": 0.0,
                "keyword_search_ms": 0.0,
                "source_scope_ms": round(scope_elapsed_ms, 2),
                "rrf_fusion_ms": 0.0,
                "hybrid_total_ms": round((perf_counter() - total_started_at) * 1000, 2),
            }

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
        fused_documents = self._reciprocal_rank_fusion(
            ranked_document_lists=[vector_documents, keyword_documents],
            top_k=len(vector_documents) + len(keyword_documents),
        )
        # 重排前保留各路靠前候选，避免单一路线的答案片段在截断时消失。
        documents = self._diversify_candidates(
            fused_documents,
            vector_documents,
            keyword_documents,
            top_k,
        )
        fusion_elapsed_ms = (perf_counter() - fusion_started_at) * 1000

        return documents, {
            "vector_search_ms": round(vector_elapsed_ms, 2),
            "keyword_search_ms": round(keyword_elapsed_ms, 2),
            "source_scope_ms": round(scope_elapsed_ms, 2),
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
        return self._rank_stored_documents(
            query, stored_data, settings.retrieval_keyword_top_k
        )

    def _explicit_source_search(
        self, query: str, source_hint: str | None = None, top_k: int = 6
    ) -> list[Document]:
        """问题明确只指定一门课的一周/一讲时，补充该资料的向量候选。"""
        hint = source_hint if source_hint is not None else query
        courses = {code.upper() for code in re.findall(r"\b[A-Za-z]{3,4}\d{4}\b", hint)}
        references = {
            (kind.lower(), int(number))
            for kind, number in re.findall(r"\b(week|lecture)\s*0*(\d{1,2})\b", hint, re.I)
        }
        references.update(
            ("week" if kind == "周" else "lecture", int(number))
            for number, kind in re.findall(r"第\s*(\d{1,2})\s*(周|讲)", hint)
        )
        if len(courses) != 1 or len(references) != 1:
            return []

        course = next(iter(courses))
        kind, number = next(iter(references))
        prefix = re.compile(rf"^{course}_{kind}0*{number}_", re.I)
        stored_data = self.vector_store.get(include=["metadatas"])
        sources = sorted({
            source
            for metadata in stored_data.get("metadatas", [])
            if isinstance(metadata, dict)
            if isinstance(source := metadata.get("source"), str) and prefix.match(source)
        })
        if not sources:
            return []

        source_filter = (
            {"source": sources[0]}
            if len(sources) == 1
            else {"source": {"$in": sources}}
        )
        vector_documents = self.vector_store.similarity_search(
            query=query,
            k=min(top_k, settings.retrieval_vector_top_k),
            filter=source_filter,
        )
        stored_documents = self.vector_store.get(
            where=source_filter, include=["documents", "metadatas"]
        )
        keyword_documents = self._rank_stored_documents(query, stored_documents, top_k)
        fused = self._reciprocal_rank_fusion(
            [vector_documents, keyword_documents],
            top_k=len(vector_documents) + len(keyword_documents),
        )
        vector_count = min(2, top_k // 3) if top_k > settings.retrieval_top_k else 1
        selected = {
            self._document_key(document)
            for document in [
                *vector_documents[:vector_count],
                *keyword_documents[:top_k - vector_count],
            ]
        }
        for document in fused:
            if len(selected) >= top_k:
                break
            selected.add(self._document_key(document))
        return [document for document in fused if self._document_key(document) in selected][:top_k]

    def _rank_stored_documents(
        self, query: str, stored_data: dict, top_k: int
    ) -> list[Document]:
        """只在明确指定的资料内做 BM25，和限定范围的向量召回互补。"""
        documents = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(
                stored_data.get("documents") or [],
                stored_data.get("metadatas") or [],
            )
            if text
        ]
        if not documents:
            return []
        tokenized = [self._tokenize(document.page_content) for document in documents]
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores = BM25Okapi(tokenized).get_scores(query_tokens)
        ranking = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [
            documents[index]
            for index in ranking[:top_k]
            if scores[index] > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """把英文单词、数字和中文字符切成 BM25 可计算的 token。"""
        return [
            token
            for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())
            if token not in _ENGLISH_STOPWORDS
        ]

    @staticmethod
    def _document_key(document: Document) -> tuple[str, str]:
        """为同一文本切片生成稳定键，供两路召回结果去重。"""
        return document.page_content, str(document.metadata.get("file_id", ""))

    def _diversify_candidates(
        self,
        fused: list[Document],
        vector: list[Document],
        keyword: list[Document],
        top_k: int,
    ) -> list[Document]:
        """给重排候选保留各召回入口；直接回答的 top_k 仍按 RRF 排序。"""
        if top_k <= settings.retrieval_top_k:
            return fused[:top_k]

        per_route = min(3, top_k // 2)
        selected = {
            self._document_key(document)
            for document in [
                *vector[:per_route], *keyword[:per_route]
            ]
        }
        for document in fused:
            if len(selected) >= top_k:
                break
            selected.add(self._document_key(document))
        return [
            document for document in fused
            if self._document_key(document) in selected
        ][:top_k]

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
