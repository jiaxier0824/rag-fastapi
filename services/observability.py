"""本地 JSONL 请求日志：用于性能分析与检索结果追溯。"""

import json
from datetime import datetime, timezone
from pathlib import Path


class RequestTraceLogger:
    """每个请求写一行 JSON，避免记录回答正文、API Key 等敏感内容。"""

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)

    def write(self, event: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        enriched_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(enriched_event, ensure_ascii=False) + "\n")
