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

from config.settings import settings
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


class RagService:
    """组装“检索 -> 提示词 -> 大模型 -> 历史记录”的 RAG 问答链。"""

    def __init__(
        self,
        vector_store_service: VectorStoreService
    ):

        # 保存依赖注入的同一个向量库服务；上传和问答因此访问同一份 Chroma 数据。
        self.vector_service = vector_store_service

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
            model=settings.chat_model_name
        )

        # 4. 创建最终 RAG 链。注意 self.chain 最终是“带历史包装器的链”，不是裸 rag_chain。
        self.chain = self.__get_chain()


    def __get_chain(self):
        """
        创建完整的RAG问答链。
        """

        # 1. 取得 Chroma 检索器。这里只创建工具，不执行搜索。
        retriever = self.vector_service.get_retriever()


        # 2. 把检索到的Document列表转换成字符串
        def format_documents(
            documents: list[Document]
        ):
            # Retriever 的输出是 list[Document]；每个 Document 有 page_content 和 metadata。
            if not documents:
                return "没有检索到相关参考资料"

            # Document 来自 Chroma 检索。这里只把片段和 metadata 整理为提示词上下文。
            formatted_text = ""

            for document in documents:
                formatted_text += (
                    f"文档片段：{document.page_content}\n"
                    f"文档元数据：{document.metadata}\n\n"
                )

            return formatted_text


        # 3. 从输入字典中取出用户问题
        def format_for_retriever(value: dict):
            # 历史包装器注入后 value 同时含 input 与 history；检索器只需要本轮问题。
            return value["input"]


        # 4. 整理提示词需要的数据
        def format_for_prompt(value: dict):
            # 两条并行分支的输出在此汇合，整理为 PromptTemplate 的三个占位符。
            # 此时 value 的形状为：
            # {"input": {"input": 问题, "history": 历史消息}, "context": 检索后的字符串}
            # 返回后变为平铺字典，正好匹配模板中的 {input}、{context}、history。
            new_value = {
                "input": value["input"]["input"],
                "context": value["context"],
                "history": value["input"]["history"]
            }

            return new_value


        # 5. 创建RAG调用链
        rag_chain = (
            # input 分支保留问题与 history；context 分支只取问题并完成检索、格式化。
            # 两个分支接收同一份输入并行运行；它们的结果合并成一个字典后再进入下一节点。
            {
                "input": RunnablePassthrough(),
                "context": (
                    RunnableLambda(format_for_retriever)
                    | retriever
                    | RunnableLambda(format_documents)
                )
            }
            | RunnableLambda(format_for_prompt)
            | self.prompt_template
            # PromptTemplate 输出 ChatPromptValue；打印节点只用于观察，不改变输出。
            | RunnableLambda(print_prompt)
            # ChatTongyi 输出 AIMessage；StrOutputParser 取出其中的纯文本 answer。
            | self.chat_model
            | StrOutputParser()
        )


        # 6. 给RAG链增加长期聊天记录
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
