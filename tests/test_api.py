"""HTTP 接口的最小回归测试。

这里用假的 Service 替代 MySQL、Chroma 和大模型，验证 Router 的输入、输出、
依赖注入与状态码没有被改坏。真实三方服务的联调仍在本地或 Docker 环境完成。
"""

from datetime import datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api import app
from config.database import get_db
from dependencies import get_knowledge_file_service, get_rag_service


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


class FakeChain:
    def invoke(self, data, config):
        assert data == {"input": "课程怎么计分？"}
        assert config["configurable"]["session_id"] == "test-user"
        return "课程成绩由作业和测验组成。"


class FakeRagService:
    chain = FakeChain()


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
        json={
            "question": "课程怎么计分？",
            "session_id": "test-user",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "课程成绩由作业和测验组成。",
        "session_id": "test-user",
    }
