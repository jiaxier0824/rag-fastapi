"""检索评测指标：只判断资料来源是否被找对，不评估大模型生成答案。"""


def source_names(documents) -> list[str]:
    """按结果顺序提取来源文件名并去重。"""
    return list(
        dict.fromkeys(
            document.metadata["source"]
            for document in documents
            if document.metadata.get("source")
        )
    )


def hit_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> int:
    """Top K 中出现任一正确来源，记为 1，否则 0。"""
    return int(bool(set(retrieved_sources) & set(expected_sources)))


def reciprocal_rank(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> float:
    """正确来源第一次出现的位置的倒数；未命中则为 0。"""
    expected = set(expected_sources)
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            return 1 / rank
    return 0.0


def average(values: list[float]) -> float:
    """计算平均值，并避免空评测集产生除零错误。"""
    return round(sum(values) / len(values), 4) if values else 0.0
