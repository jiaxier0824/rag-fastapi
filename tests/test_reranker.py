"""重排服务测试：验证远程排序结果和失败降级策略。"""

from langchain_core.documents import Document

from config.settings import settings
from services.reranker import RerankService


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.7},
            ]
        }


def test_rerank_uses_remote_result_order(monkeypatch):
    documents = [
        Document(page_content="主题相关但没有答案", metadata={}),
        Document(page_content="直接回答用户问题的资料", metadata={}),
    ]
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "dashscope_workspace_id", "workspace-test")
    monkeypatch.setattr(settings, "dashscope_api_key", "test-key")

    def fake_post(url, headers, json, timeout):
        assert "workspace-test" in url
        assert json["documents"] == [
            "主题相关但没有答案",
            "直接回答用户问题的资料",
        ]
        return FakeResponse()

    monkeypatch.setattr("services.reranker.httpx.post", fake_post)

    result = RerankService().rerank("用户问题", documents)

    assert [document.page_content for document in result] == [
        "直接回答用户问题的资料",
        "主题相关但没有答案",
    ]


def test_rerank_disabled_returns_rrf_top_k(monkeypatch):
    documents = [
        Document(page_content=f"资料 {index}", metadata={})
        for index in range(5)
    ]
    monkeypatch.setattr(settings, "rerank_enabled", False)

    result = RerankService().rerank("用户问题", documents)

    assert len(result) == settings.retrieval_top_k
