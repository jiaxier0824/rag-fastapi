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
    def ask(self, question: str, session_id: str, trace_id: str):
        assert question == "课程怎么计分？"
        assert session_id == "test-user"
        assert trace_id == "agent-trace-001"
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


def test_rag_service_reuses_retrieved_documents_for_answer_and_sources():
    class FakeVectorStoreService:
        def search_with_trace(self, query: str, top_k: int):
            assert query == "课程怎么计分？"
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

    class FakeConversationChain:
        def invoke(self, data, config):
            assert data["input"] == "课程怎么计分？"
            assert len(data["documents"]) == 2
            assert config["configurable"]["session_id"] == "test-user"
            return "课程成绩由作业和测验组成。"

    rag_service = object.__new__(RagService)
    rag_service.vector_service = FakeVectorStoreService()
    rag_service.chain = FakeConversationChain()

    class FakeRerankService:
        def rerank(self, question: str, candidates):
            assert question == "课程怎么计分？"
            assert len(candidates) == 2
            return candidates

    rag_service.rerank_service = FakeRerankService()

    class FakeTraceLogger:
        def write(self, event):
            assert event["trace_id"] == "trace-test"
            assert event["sources"] == ["assessment.md"]

    rag_service.trace_logger = FakeTraceLogger()

    answer, source_filenames = rag_service.ask(
        question="课程怎么计分？",
        session_id="test-user",
        trace_id="trace-test",
    )

    assert answer == "课程成绩由作业和测验组成。"
    assert source_filenames == ["assessment.md"]
