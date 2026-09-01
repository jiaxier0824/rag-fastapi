"""RAG 问答链的组装。

链路可以读成：用户问题 -> Chroma 检索相关切片 -> 组装 context/history/input 提示词
-> ChatTongyi 生成 -> 字符串答案；最外层再由 RunnableWithMessageHistory 负责读写历史。
"""

from langchain_core.documents import Document
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


def print_prompt(prompt):
    """
    在终端中打印大模型最终收到的提示词。
    主要用于学习和排错。它会原样返回 prompt，所以不会中断 LCEL 管道。
    """

    print("=" * 50)
    print(prompt.to_string())
    print("=" * 50)

    return prompt


def _source_names(documents: list[Document]) -> list[str]:
    """按顺序提取检索或重排阶段出现的来源文件名。"""
    return list(
        dict.fromkeys(
            document.metadata["source"]
            for document in documents
            if document.metadata.get("source")
        )
    )


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
                    "请以提供的参考资料为主，"
                    "简洁、专业地回答用户问题。"
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
        )

        # 4. 创建最终 RAG 链。注意 self.chain 最终是“带历史包装器的链”，不是裸 rag_chain。
        self.chain = self.__get_chain()

    def ask(
        self,
        question: str,
        session_id: str,
        trace_id: str,
    ) -> tuple[str, list[str]]:
        """完成一次问答，并返回答案和本次实际使用的来源文件名。"""
        # 只检索一次：同一批 Document 同时用作模型上下文和前端来源。
        candidate_top_k = (
            settings.retrieval_candidate_top_k
            if settings.rerank_enabled
            else settings.retrieval_top_k
        )
        candidates, retrieval_timing = self.vector_service.search_with_trace(
            query=question,
            top_k=candidate_top_k,
        )

        rerank_started_at = perf_counter()
        documents = self.rerank_service.rerank(
            question=question,
            candidates=candidates,
        )
        rerank_elapsed_ms = round((perf_counter() - rerank_started_at) * 1000, 2)

        source_filenames = list(
            dict.fromkeys(
                document.metadata["source"]
                for document in documents
                if document.metadata.get("source")
            )
        )

        session_config = {
            "configurable": {
                "session_id": session_id
            }
        }

        generation_started_at = perf_counter()
        answer = self.chain.invoke(
            {
                "input": question,
                "documents": documents,
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
                "rerank_enabled": settings.rerank_enabled,
                "candidate_count": len(candidates),
                "final_document_count": len(documents),
                "candidate_sources": _source_names(candidates),
                "sources": source_filenames,
                "timing_ms": {
                    **retrieval_timing,
                    "rerank_ms": rerank_elapsed_ms,
                    "generation_ms": generation_elapsed_ms,
                    "total_ms": round(
                        retrieval_timing["hybrid_total_ms"]
                        + rerank_elapsed_ms
                        + generation_elapsed_ms,
                        2,
                    ),
                },
            }
        )

        return answer, source_filenames

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
                "history": value["input"]["history"]
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
            # PromptTemplate 输出 ChatPromptValue；打印节点只用于观察，不改变输出。
            | RunnableLambda(print_prompt)
            # ChatTongyi 输出 AIMessage；StrOutputParser 取出其中的纯文本 answer。
            | self.chat_model
            | StrOutputParser()
        )


        conversation_chain = RunnableWithMessageHistory(
            # 包装器会：1) 按 session_id 读取历史；2) 注入 history；3) 自动保存本轮问答。
            # input_messages_key 告诉它“用户消息在输入字典的哪个键”；
            # history_messages_key 告诉它“把读取到的历史注入到哪个键”。
            rag_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )

        return conversation_chain
