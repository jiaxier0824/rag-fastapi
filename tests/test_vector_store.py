"""混合检索的纯单元测试，不连接真实 Chroma 或 Embedding 服务。"""

import pytest

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from knowledge_base import KnowledgeBaseService
from stores.vector_store import VectorStoreService


def test_context_window_keeps_real_neighbor_without_adding_candidates(tmp_path, monkeypatch):
    file_id = "11111111-1111-4111-8111-111111111111"
    text = "Topic: weekly quizzes.\n\n" + " ".join(
        f"background{i}" for i in range(200)
    ) + "\n\nBest results count."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=settings.separators,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    assert len(chunks) >= 3
    (tmp_path / f"{file_id}.md").write_text(text, encoding="utf-8")
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))

    class FakeWindowChroma:
        def get(self, where, include):
            assert where == {"file_id": file_id}
            assert include == ["documents", "metadatas"]
            return {"documents": chunks, "metadatas": [{} for _ in chunks]}

    service = object.__new__(VectorStoreService)
    service.vector_store = FakeWindowChroma()
    candidate = Document(page_content=chunks[0], metadata={"file_id": file_id})
    adjacent = Document(page_content=chunks[1], metadata={"file_id": file_id})

    windows = service.expand_context_windows([candidate, adjacent])

    assert len(windows) == 1
    assert windows[0].page_content == "\n".join(chunks[:3])
    assert windows[0].metadata == candidate.metadata


def test_context_window_does_not_guess_when_index_differs(tmp_path, monkeypatch):
    file_id = "22222222-2222-4222-8222-222222222222"
    (tmp_path / f"{file_id}.md").write_text("original", encoding="utf-8")
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))

    class FakeWindowChroma:
        def get(self, where, include):
            return {"documents": ["stale", "neighbor"], "metadatas": [{}, {}]}

    service = object.__new__(VectorStoreService)
    service.vector_store = FakeWindowChroma()
    candidate = Document(page_content="stale", metadata={"file_id": file_id})

    assert service.expand_context_windows([candidate]) == [candidate]


def test_context_window_uses_chunk_index_without_markdown_original():
    file_id = "33333333-3333-4333-8333-333333333333"

    class IndexedChroma:
        def get(self, where, include):
            return {
                "documents": ["second", "first"],
                "metadatas": [{"chunk_index": 1}, {"chunk_index": 0}],
            }

    service = object.__new__(VectorStoreService)
    service.vector_store = IndexedChroma()
    candidate = Document(page_content="first", metadata={"file_id": file_id})

    assert service.expand_context_windows([candidate])[0].page_content == "first\nsecond"


def test_new_chunks_store_their_order_in_metadata():
    class RecordingVectorStore:
        def add_texts(self, texts, metadatas):
            assert [metadata["chunk_index"] for metadata in metadatas] == list(range(len(texts)))
            return [f"id-{index}" for index in range(len(texts))]

    knowledge_base = KnowledgeBaseService(RecordingVectorStore())
    assert len(knowledge_base.add_texts("unique " * 300, "course.md")) >= 2


def test_bm25_tokenizer_removes_english_stopwords_but_keeps_search_terms():
    assert VectorStoreService._tokenize(
        "What problems if REIT6811 research does not engage users?"
    ) == ["problems", "reit6811", "research", "not", "engage", "users"]


@pytest.mark.parametrize("scoped", [False, True], ids=["global", "course-week"])
@pytest.mark.parametrize(
    "vector_names,keyword_names,top_k,expected",
    [
        # B 在两路排第2、第3，RRF分数最高；不能被两路配额排除。
        (["A", "B", "C"], ["D", "E", "B"], 3, ["B", "A", "D"]),
        # E/F 在两路都靠后，但累计贡献最高，Top-6也必须尊重融合排名。
        (list("ABCDEF"), list("GHIJEF"), 6, ["E", "F", "A", "G", "B", "H"]),
    ],
    ids=["top3-keeps-best-shared-chunk", "top6-keeps-best-shared-chunks"],
)
def test_hybrid_candidates_follow_rrf_top_k(
    monkeypatch, scoped, vector_names, keyword_names, top_k, expected
):
    monkeypatch.setattr(settings, "retrieval_rrf_k", 60)
    monkeypatch.setattr(settings, "retrieval_top_k", 3)
    monkeypatch.setattr(settings, "retrieval_vector_top_k", 10)
    source = "INFS7410_week1_notes.md"
    metadata = {"source": source, "file_id": "course-file"}
    vector = [Document(page_content=name, metadata=metadata) for name in vector_names]
    keyword = [Document(page_content=name, metadata=metadata) for name in keyword_names]

    class RankedChroma:
        def similarity_search(self, *, query, k, filter=None):
            assert filter == ({"source": source} if scoped else None)
            return vector[:k]

        def get(self, *, include, where=None):
            if include == ["metadatas"]:
                return {"metadatas": [metadata]}
            assert where == ({"source": source} if scoped else None)
            return {"documents": keyword_names, "metadatas": [metadata] * len(keyword_names)}

    service = object.__new__(VectorStoreService)
    service.vector_store = RankedChroma()
    # 两路已有排名固定，只验证融合后的候选选择，不依赖BM25分词或模型。
    monkeypatch.setattr(
        service, "_rank_stored_documents",
        lambda query, stored_data, limit: keyword[:limit],
    )
    documents, _ = service.search_with_trace(
        query="assignment deadline",
        source_hint="INFS7410 第1周" if scoped else "作业什么时候截止？",
        top_k=top_k,
    )

    assert [document.page_content for document in documents] == expected
    assert len({service._document_key(document) for document in documents}) == len(documents)
    assert all(document.metadata == metadata for document in documents)


class FakeSourceChroma:
    def get(self, include: list[str], where: dict | None = None):
        if where is not None:
            assert where == {"source": "DECO6500_Week01_Introduction.md"}
            assert include == ["documents", "metadatas"]
            return {"documents": ["A1, A2, A3"], "metadatas": [where]}
        assert include == ["metadatas"]
        return {"metadatas": [
            {"source": "DECO6500_Week01_Introduction.md"},
            {"source": "DECO6500_Week09_Review.md"},
            {"source": "INFS7410_Week01_Introduction.md"},
        ]}

    def similarity_search(self, query: str, k: int, filter: dict):
        assert k == 6
        assert filter == {"source": "DECO6500_Week01_Introduction.md"}
        return [Document(page_content="A1, A2, A3", metadata={"source": filter["source"]})]


def test_explicit_course_week_adds_scoped_candidates_without_hardcoding_a_file():
    service = object.__new__(VectorStoreService)
    service.vector_store = FakeSourceChroma()

    documents = service._explicit_source_search("DECO6500 第 1 周的 A1、A2、A3 是什么？")

    assert [document.page_content for document in documents] == ["A1, A2, A3"]


def test_explicit_source_does_not_mix_other_weeks_back_into_candidates():
    service = object.__new__(VectorStoreService)
    service.vector_store = FakeSourceChroma()

    documents, timing = service.search_with_trace(
        query="DECO6500 Week 1 A1 A2 A3",
        top_k=6,
        source_hint="DECO6500 第 1 周的 A1、A2、A3 是什么？",
    )

    assert [document.page_content for document in documents] == ["A1, A2, A3"]
    assert timing["source_scope_ms"] >= 0


def test_cross_week_question_does_not_limit_sources():
    service = object.__new__(VectorStoreService)
    service.vector_store = FakeSourceChroma()

    assert service._explicit_source_search("DECO6500 Week 1 和 Week 9 有什么关系？") == []


def test_query_rewrite_cannot_invent_a_source_scope():
    service = object.__new__(VectorStoreService)
    service.vector_store = FakeSourceChroma()

    assert service._explicit_source_search(
        "DECO6500 Week 1 assignments", source_hint="DECO6500 的作业有哪些？"
    ) == []


class FakeChroma:
    def similarity_search(self, query: str, k: int):
        assert query == "INFS7410 作业截止时间"
        assert k == settings.retrieval_vector_top_k
        return [
            Document(
                page_content="课程作业会在每周发布。",
                metadata={"file_id": "course"},
            ),
        ]

    def get(self, include: list[str]):
        assert include == ["documents", "metadatas"]
        return {
            "documents": [
                "课程作业会在每周发布。",
                "INFS7410 Project 1 截止时间为 9 月 15 日。",
                "本课程也包含课堂讨论和阅读材料。",
            ],
            "metadatas": [
                {"file_id": "course"},
                {"file_id": "assessment"},
                {"file_id": "reading"},
            ],
        }


def test_hybrid_search_merges_vector_and_keyword_results():
    service = object.__new__(VectorStoreService)
    service.vector_store = FakeChroma()

    documents, _ = service.search_with_trace(
        query="INFS7410 作业截止时间",
        top_k=settings.retrieval_top_k,
    )

    assert [document.page_content for document in documents] == [
        "课程作业会在每周发布。",
        "INFS7410 Project 1 截止时间为 9 月 15 日。",
    ]
