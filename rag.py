"""RAG 问答链的组装。

链路可以读成：用户问题 -> Chroma 检索相关切片 -> 组装 context/history/input 提示词
-> ChatTongyi 生成 -> 字符串答案；最外层再由 RunnableWithMessageHistory 负责读写历史。
"""

import logging
import re
from hashlib import sha256

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableWithMessageHistory,
    RunnableLambda
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_community.chat_models.tongyi import ChatTongyi
from time import perf_counter

from config.settings import settings
from services.reranker import RerankService
from services.observability import RequestTraceLogger
from stores.chat_history import get_history
from stores.vector_store import VectorStoreService


logger = logging.getLogger(__name__)

def _source_names(documents: list[Document]) -> list[str]:
    """按顺序提取检索或重排阶段出现的来源文件名。"""
    return list(
        dict.fromkeys(
            document.metadata["source"]
            for document in documents
            if document.metadata.get("source")
        )
    )


def _chunk_refs(documents: list[Document]) -> list[dict[str, str]]:
    """记录可回查的切片指纹，不把课件正文写进请求日志。"""
    return [
        {
            "source": str(document.metadata.get("source", "")),
            "content_sha256": sha256(document.page_content.encode("utf-8")).hexdigest(),
        }
        for document in documents
    ]


class RagService:
    """组装“检索 -> 提示词 -> 大模型 -> 历史记录”的 RAG 问答链。"""

    def __init__(
        self,
        vector_store_service: VectorStoreService,
        rerank_service: RerankService,
        trace_logger: RequestTraceLogger,
    ):

        # 保存依赖注入的同一个向量库服务；上传和问答因此访问同一份 Chroma 数据。
        self.vector_service = vector_store_service
        self.rerank_service = rerank_service
        self.trace_logger = trace_logger

        # 2. 创建聊天提示词。三个占位符会在链运行时填入：context、history、input。
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "只能依据提供的参考资料回答；资料没有明确说明时，直接说明未找到，不能猜测或补全。"
                    "涉及日期、时间、分数、百分比、权重或编号时，必须从资料中逐字提取并保留原始格式，"
                    "不得自行换算、转换日期顺序或修改数值。"
                    "当用户明确要求保留某个术语、缩写或英文表达时，答案必须原样包含该表达。"
                    "若多段资料有冲突，优先采用课程 Profile 或明确的官方评估信息。"
                    "回答前逐一检查所有参考片段，合并直接相关且互补的事实，避免只回答问题的一部分。"
                    "回答应简洁、专业。"
                    "参考资料：{context}"
                ),
                (
                    "system",
                    "下面是用户以前的对话记录："
                ),
                MessagesPlaceholder(
                    variable_name="history"
                ),
                (
                    "user",
                    "请回答用户问题：{input}"
                )
            ]
        )

        # 3. 创建聊天模型
        self.chat_model = ChatTongyi(
            model=settings.chat_model_name,
            api_key=settings.dashscope_api_key,
            temperature=settings.chat_temperature,
        )

        # 4. 同时保留无历史的基础链和面向独立 RAG 对话的历史包装链。
        self.rag_chain = self.__get_chain()
        self.chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def ask(
        self,
        question: str,
        session_id: str,
        trace_id: str,
        use_history: bool = True,
    ) -> tuple[str, list[str]]:
        """完成一次问答，并返回答案和本次实际使用的来源文件名。"""
        # 只检索一次：同一批 Document 同时用作模型上下文和前端来源。
        candidate_top_k = (
            settings.retrieval_candidate_top_k
            if settings.rerank_enabled
            else settings.retrieval_top_k
        )
        rewrite_started_at = perf_counter()
        retrieval_query = self._retrieval_query(question)
        rewrite_elapsed_ms = round((perf_counter() - rewrite_started_at) * 1000, 2)
        candidates, retrieval_timing = self.vector_service.search_with_trace(
            query=retrieval_query,
            top_k=candidate_top_k,
            source_hint=question,
        )
        candidates = self.vector_service.expand_context_windows(candidates)

        rerank_started_at = perf_counter()
        documents = self.rerank_service.rerank(
            question=retrieval_query,
            candidates=candidates,
        )
        rerank_elapsed_ms = round((perf_counter() - rerank_started_at) * 1000, 2)

        source_filenames = _source_names(documents)

        session_config = {
            "configurable": {
                "session_id": session_id
            }
        }

        generation_started_at = perf_counter()
        chain = self.chain if use_history else self.rag_chain
        answer = chain.invoke(
            {
                "input": question,
                "documents": documents,
                "history": [],
            },
            config=session_config,
        )
        generation_elapsed_ms = round(
            (perf_counter() - generation_started_at) * 1000,
            2,
        )

        self.trace_logger.write(
            {
                "trace_id": trace_id,
                "session_id": session_id,
                "question_length": len(question),
                "model": settings.chat_model_name,
                "query_rewritten": retrieval_query != question,
                "rerank_enabled": settings.rerank_enabled,
                "candidate_count": len(candidates),
                "final_document_count": len(documents),
                "candidate_sources": _source_names(candidates),
                "candidate_chunks": _chunk_refs(candidates),
                "sources": source_filenames,
                "final_chunks": _chunk_refs(documents),
                "timing_ms": {
                    "query_rewrite_ms": rewrite_elapsed_ms,
                    **retrieval_timing,
                    "rerank_ms": rerank_elapsed_ms,
                    "generation_ms": generation_elapsed_ms,
                    "total_ms": round(
                        rewrite_elapsed_ms
                        + retrieval_timing["hybrid_total_ms"]
                        + rerank_elapsed_ms
                        + generation_elapsed_ms,
                        2,
                    ),
                },
            }
        )

        return answer, source_filenames

    def _retrieval_query(self, question: str) -> str:
        """仅把中文问题翻译成英文检索词；生成答案仍使用原始问题。"""
        if re.search(r"[\u4e00-\u9fff]", question) is None:
            return question
        try:
            response = self.chat_model.invoke([
                SystemMessage(content=(
                    "Translate the user's Chinese question into a concise English search query "
                    "for English course documents. Preserve course codes, week/lecture numbers, "
                    "named entities, document type (such as presentation versus report), and "
                    "any numbers already in the question. Include standard English synonyms for "
                    "the question's key terms, but do not add an answer. Do not answer the "
                    "question or add facts. Output only the English query."
                )),
                HumanMessage(content=question),
            ])
            translated = response.content.strip() if isinstance(response.content, str) else ""
            course_codes = re.findall(r"\b[A-Za-z]{3,4}\d{4}\b", question)
            if not translated:
                return question
            return f"{' '.join(course_codes)} {translated}".strip()
        except Exception:
            logger.exception("检索问题翻译失败，回退到原始问题")
            return question

    def __get_chain(self):
        """
        创建完整的RAG问答链。
        """

        # 把检索到的 Document 列表转换成提示词中的 context 字符串。
        def format_documents(
            documents: list[Document]
        ):
            if not documents:
                return "没有检索到相关参考资料"

            formatted_text = ""

            for document in documents:
                formatted_text += (
                    f"文档片段：{document.page_content}\n"
                    f"文档元数据：{document.metadata}\n\n"
                )

            return formatted_text

        # 两条并行分支汇合后，整理为 PromptTemplate 所需的三个占位符。
        def format_for_prompt(value: dict):
            return {
                "input": value["input"]["input"],
                "context": value["context"],
                "history": value["input"]["history"],
            }

        rag_chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(
                    lambda value: format_documents(value["documents"])
                ),
            }
            | RunnableLambda(format_for_prompt)
            | self.prompt_template
            # ChatTongyi 输出 AIMessage；StrOutputParser 取出其中的纯文本 answer。
            | self.chat_model
            | StrOutputParser()
        )


        return rag_chain
