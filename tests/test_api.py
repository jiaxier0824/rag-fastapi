"""HTTP 接口的最小回归测试。

这里用假的 Service 替代 MySQL、Chroma 和大模型，验证 Router 的输入、输出、
依赖注入与状态码没有被改坏。真实三方服务的联调仍在本地或 Docker 环境完成。
"""

from datetime import datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api import app
from config.database import get_db
from config.settings import settings
from dependencies import get_knowledge_file_service, get_rag_service
from rag import RagService
from stores.chat_history import FileChatMessageHistory


def test_chinese_retrieval_query_is_translated_without_changing_original_question():
    class FakeModel:
        def invoke(self, messages):
            assert messages[-1].content == "INFS7203 尿布到啤酒的支持度和置信度？"
            return SimpleNamespace(content="association rule diapers to beer support and confidence")

    rag_service = object.__new__(RagService)
    rag_service.chat_model = FakeModel()
    assert rag_service._retrieval_query("INFS7203 尿布到啤酒的支持度和置信度？") == (
        "INFS7203 association rule diapers to beer support and confidence"
    )
    assert rag_service._retrieval_query("INFS7203 support and confidence") == (
        "INFS7203 support and confidence"
    )


class FakeKnowledgeFileService:
    def __init__(self):
        self.record = SimpleNamespace(
            file_id="file-test-001",
            filename="course.md",
            content_type="text/markdown",
            file_size=18,
            status="completed",
            chunk_count=1,
            error_message=None,
            created_at=datetime(2026, 8, 7, 10, 0, 0),
            updated_at=datetime(2026, 8, 7, 10, 0, 0),
        )

    async def upload_file(self, **_kwargs):
        return self.record, True

    async def list_files(self, **_kwargs):
        return [self.record], 1

    async def delete_file(self, **kwargs):
        if kwargs["file_id"] == self.record.file_id:
            return self.record, 1
        return None, 0


class FakeRagService:
    def ask(self, question: str, session_id: str, trace_id: str, use_history: bool = True):
        assert question == "课程怎么计分？"
        assert session_id == "test-user"
        assert trace_id == "agent-trace-001"
        assert use_history is True
        return "课程成绩由作业和测验组成。", ["course.md"]


async def fake_get_db():
    """接口测试不触发真实 MySQL 连接。"""
    yield object()


def build_client() -> TestClient:
    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_knowledge_file_service] = FakeKnowledgeFileService
    app.dependency_overrides[get_rag_service] = FakeRagService
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_health_check():
    response = build_client().get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_list_and_delete_knowledge_file():
    client = build_client()

    upload_response = client.post(
        "/api/knowledge/upload",
        files={"file": ("course.md", "# Course\ncontent", "text/markdown")},
    )
    assert upload_response.status_code == 200
    assert upload_response.json()["filename"] == "course.md"

    list_response = client.get("/api/knowledge/files?page=1&page_size=20")
    assert list_response.status_code == 200
    assert list_response.json()["total"] == 1
    assert list_response.json()["items"][0]["file_id"] == "file-test-001"

    delete_response = client.delete("/api/knowledge/files/file-test-001")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted_chunks"] == 1


def test_upload_rejects_unsupported_file_type():
    response = build_client().post(
        "/api/knowledge/upload",
        files={"file": ("course.pdf", b"not parsed", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "目前只支持txt和md文件"


def test_chat_uses_request_question_and_session_id():
    response = build_client().post(
        "/api/rag/chat",
        headers={"X-Trace-ID": "agent-trace-001"},
        json={
            "question": "课程怎么计分？",
            "session_id": "test-user",
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "课程成绩由作业和测验组成。"
    assert response.json()["session_id"] == "test-user"
    assert response.json()["sources"] == [{"filename": "course.md"}]
    assert response.json()["trace_id"] == "agent-trace-001"


def test_agent_can_disable_rag_history():
    class HistoryAwareFakeRagService(FakeRagService):
        def ask(self, question: str, session_id: str, trace_id: str, use_history: bool = True):
            assert use_history is False
            return "无重复历史的答案", ["course.md"]

    app.dependency_overrides[get_rag_service] = HistoryAwareFakeRagService
    response = TestClient(app).post(
        "/api/rag/chat",
        headers={"X-Trace-ID": "agent-trace-001"},
        json={"question": "课程怎么计分？", "session_id": "test-user", "use_history": False},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "无重复历史的答案"


def test_chat_rejects_unsafe_session_id():
    response = build_client().post(
        "/api/rag/chat",
        json={"question": "课程怎么计分？", "session_id": "../../outside"},
    )

    assert response.status_code == 422


def test_history_store_rejects_unsafe_session_id(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="session_id"):
        FileChatMessageHistory("../../outside", str(tmp_path))


def test_rag_service_reuses_retrieved_documents_for_answer_and_sources():
    class FakeVectorStoreService:
        def search_with_trace(self, query: str, top_k: int, source_hint: str | None = None):
            assert query == "course assessment grading"
            expected_top_k = (
                settings.retrieval_candidate_top_k
                if settings.rerank_enabled
                else settings.retrieval_top_k
            )
            assert top_k == expected_top_k
            return [
                Document(
                    page_content="作业占 60%。",
                    metadata={"source": "assessment.md"},
                ),
                Document(
                    page_content="测验占 40%。",
                    metadata={"source": "assessment.md"},
                ),
            ], {"hybrid_total_ms": 1.0}

        def expand_context_windows(self, candidates):
            return candidates

    class FakeConversationChain:
        def invoke(self, data, config):
            assert data["input"] == "课程怎么计分？"
            assert len(data["documents"]) == 2
            assert config["configurable"]["session_id"] == "test-user"
            return "课程成绩由作业和测验组成。"

    rag_service = object.__new__(RagService)
    rag_service._retrieval_query = lambda question: "course assessment grading"
    rag_service.vector_service = FakeVectorStoreService()
    rag_service.chain = FakeConversationChain()
    rag_service.rag_chain = FakeConversationChain()

    class FakeRerankService:
        def rerank(self, question: str, candidates):
            assert question == "course assessment grading"
            assert len(candidates) == 2
            return candidates

    rag_service.rerank_service = FakeRerankService()

    class FakeTraceLogger:
        def write(self, event):
            assert event["trace_id"] == "trace-test"
            assert event["sources"] == ["assessment.md"]
            assert event["query_rewritten"] is True
            assert len(event["candidate_chunks"]) == 2
            assert len(event["final_chunks"]) == 2
            assert event["final_chunks"][0]["source"] == "assessment.md"
            assert len(event["final_chunks"][0]["content_sha256"]) == 64
            assert "page_content" not in event["final_chunks"][0]

    rag_service.trace_logger = FakeTraceLogger()

    answer, source_filenames = rag_service.ask(
        question="课程怎么计分？",
        session_id="test-user",
        trace_id="trace-test",
    )

    assert answer == "课程成绩由作业和测验组成。"
    assert source_filenames == ["assessment.md"]


def test_rag_service_uses_stateless_chain_when_history_is_disabled():
    document = Document(page_content="作业占 60%。", metadata={"source": "assessment.md"})

    class FakeVectorStoreService:
        def search_with_trace(self, **_kwargs):
            return [document], {"hybrid_total_ms": 1.0}

        def expand_context_windows(self, candidates):
            return candidates

    class FakeRerankService:
        def rerank(self, **_kwargs):
            return [document]

    class StatelessChain:
        def invoke(self, data, config):
            assert data["history"] == []
            return "无历史污染的答案"

    class ConversationChain:
        def invoke(self, *_args, **_kwargs):
            raise AssertionError("Agent 调用不应进入 RAG 会话历史链")

    class FakeTraceLogger:
        def write(self, _event):
            return None

    rag_service = object.__new__(RagService)
    rag_service._retrieval_query = lambda question: question
    rag_service.vector_service = FakeVectorStoreService()
    rag_service.rerank_service = FakeRerankService()
    rag_service.rag_chain = StatelessChain()
    rag_service.chain = ConversationChain()
    rag_service.trace_logger = FakeTraceLogger()

    answer, sources = rag_service.ask(
        question="课程怎么计分？",
        session_id="agent-session",
        trace_id="agent-trace",
        use_history=False,
    )

    assert answer == "无历史污染的答案"
    assert sources == ["assessment.md"]
