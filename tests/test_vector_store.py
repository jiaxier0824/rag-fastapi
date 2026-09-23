"""混合检索的纯单元测试，不连接真实 Chroma 或 Embedding 服务。"""

from langchain_core.documents import Document

from config.settings import settings
from stores.vector_store import VectorStoreService


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
