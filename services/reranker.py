"""百炼 qwen3-rerank 的轻量客户端，负责候选切片的二次精排。"""

import logging
import httpx
from langchain_core.documents import Document

from config.settings import settings


logger = logging.getLogger(__name__)


class RerankService:
    """对混合检索的候选 Document 进行二次排序。

    未启用、缺少业务空间 ID 或远程调用失败时，安全降级为 RRF 已排序的前 top_k 条。
    """

    def rerank(
        self,
        question: str,
        candidates: list[Document],
    ) -> list[Document]:
        fallback_documents = candidates[:settings.retrieval_top_k]

        if not settings.rerank_enabled or not candidates:
            return fallback_documents

        api_key = settings.dashscope_api_key
        workspace_id = settings.dashscope_workspace_id
        if not api_key or not workspace_id:
            logger.warning(
                "Rerank 未配置 API Key 或 Workspace ID，已降级为 RRF 结果"
            )
            return fallback_documents

        endpoint = (
            f"https://{workspace_id}.{settings.dashscope_region}.maas.aliyuncs.com/"
            "compatible-api/v1/reranks"
        )
        payload = {
            "model": settings.rerank_model_name,
            "query": question,
            "documents": [
                document.page_content
                for document in candidates
            ],
            "top_n": min(settings.rerank_top_n, len(candidates)),
            "instruct": (
                "Given a user question, rank passages by how directly they "
                "answer the question."
            ),
        }

        try:
            response = httpx.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=settings.rerank_timeout_seconds,
            )
            response.raise_for_status()
            result_items = response.json().get("results", [])

            reranked_documents = [
                candidates[item["index"]]
                for item in result_items
                if isinstance(item.get("index"), int)
                and 0 <= item["index"] < len(candidates)
            ]
            return reranked_documents or fallback_documents
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            logger.warning(
                "Rerank 调用失败，已降级为 RRF 结果：%s",
                error,
            )
            return fallback_documents
