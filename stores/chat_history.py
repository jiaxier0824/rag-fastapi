"""本地 JSON 聊天历史实现。

本项目没有把多轮历史存进 MySQL，而是按 session_id 存在 chat_history/ 下的文件中。
这适合学习项目；生产环境通常会改为数据库或 Redis，并加入用户鉴权与过期策略。
"""

import json
import re
from pathlib import Path
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict
)

from config.settings import settings


def get_history(session_id: str) -> "FileChatMessageHistory":
    """根据会话编号取得对应的聊天历史对象。

    RunnableWithMessageHistory 会调用此函数；同一个 session_id 对应同一个历史文件。
    """

    return FileChatMessageHistory(
        session_id=session_id,
        storage_path=settings.chat_history_dir
    )


class FileChatMessageHistory(BaseChatMessageHistory):
    """把 LangChain 消息序列化为本地 JSON 文件的历史记录实现。

    继承 ``BaseChatMessageHistory`` 后，本类满足 LangChain 历史组件所需的标准接口：
    ``messages`` 读取记录、``add_messages`` 追加记录、``clear`` 清空记录。
    """
    def __init__(self, session_id: str, storage_path: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise ValueError("session_id 只能包含字母、数字、下划线或连字符")
        self.session_id = session_id
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_path / self.session_id

    @property
    def messages(self) -> list[BaseMessage]:
        # 首次对话时文件不存在，返回空历史是正常情况。
        try:
            with self.file_path.open(
                "r",
                encoding="utf-8"
            ) as file:
                message_dicts = json.load(file)

            return messages_from_dict(message_dicts)
        except FileNotFoundError:
            return []

    def add_messages(
        self,
        messages: Sequence[BaseMessage]
    ) -> None:
        # 先读旧历史、追加本轮 human/AI 消息，再整体写回 JSON 文件。
        all_messages = list(self.messages)
        all_messages.extend(messages)

        message_dicts = [
            message_to_dict(message)
            for message in all_messages
        ]

        with self.file_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                message_dicts,
                file,
                ensure_ascii=False,
                indent=2
            )

    def clear(self) -> None:
        """保留历史文件，但将其中的消息数组重置为空。"""
        with self.file_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump([], file)
