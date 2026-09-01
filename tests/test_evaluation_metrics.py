from langchain_core.documents import Document

from evaluation.metrics import hit_at_k, reciprocal_rank, source_names


def test_metrics_use_first_correct_source_rank():
    documents = [
        Document(page_content="first", metadata={"source": "wrong.md"}),
        Document(page_content="second", metadata={"source": "right.md"}),
        Document(page_content="third", metadata={"source": "right.md"}),
    ]

    sources = source_names(documents)

    assert sources == ["wrong.md", "right.md"]
    assert hit_at_k(sources, ["right.md"]) == 1
    assert reciprocal_rank(sources, ["right.md"]) == 0.5
