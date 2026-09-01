"""执行混合检索基线与 qwen3-rerank 的离线 A/B 评测。"""

import json
from pathlib import Path
from time import perf_counter

from config.settings import settings
from dependencies import get_rerank_service, get_vector_store_service
from evaluation.metrics import average, hit_at_k, reciprocal_rank, source_names


EVALUATION_DIR = Path(__file__).parent
DATASET_PATH = EVALUATION_DIR / "dataset.json"
REPORT_PATH = EVALUATION_DIR / "report.json"


def run_evaluation() -> dict:
    """对相同题集跑两次排序，写出可用于简历复盘的结构化报告。"""
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    vector_store = get_vector_store_service()
    reranker = get_rerank_service()
    case_reports: list[dict] = []

    for case in cases:
        # 始终召回 6 条候选，保证 A/B 的输入完全相同。
        candidates, retrieval_timing = vector_store.search_with_trace(
            query=case["question"],
            top_k=settings.retrieval_candidate_top_k,
        )

        baseline_documents = candidates[:settings.retrieval_top_k]
        baseline_sources = source_names(baseline_documents)

        rerank_started_at = perf_counter()
        reranked_documents = reranker.rerank(
            question=case["question"],
            candidates=candidates,
        )
        rerank_elapsed_ms = round((perf_counter() - rerank_started_at) * 1000, 2)
        reranked_sources = source_names(reranked_documents)

        case_reports.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_sources": case["expected_sources"],
                "candidate_sources": source_names(candidates),
                "baseline": _result_metrics(
                    sources=baseline_sources,
                    expected_sources=case["expected_sources"],
                    latency_ms=retrieval_timing["hybrid_total_ms"],
                ),
                "rerank": _result_metrics(
                    sources=reranked_sources,
                    expected_sources=case["expected_sources"],
                    latency_ms=round(
                        retrieval_timing["hybrid_total_ms"] + rerank_elapsed_ms,
                        2,
                    ),
                ),
                "timing_ms": {
                    **retrieval_timing,
                    "rerank_ms": rerank_elapsed_ms,
                },
            }
        )

    report = {
        "dataset_size": len(case_reports),
        "retrieval_config": {
            "vector_top_k": settings.retrieval_vector_top_k,
            "keyword_top_k": settings.retrieval_keyword_top_k,
            "rrf_candidate_top_k": settings.retrieval_candidate_top_k,
            "final_top_k": settings.retrieval_top_k,
            "rerank_model": settings.rerank_model_name,
        },
        "baseline": _summary(case_reports, "baseline"),
        "rerank": _summary(case_reports, "rerank"),
        "cases": case_reports,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _result_metrics(
    sources: list[str],
    expected_sources: list[str],
    latency_ms: float,
) -> dict:
    return {
        "sources": sources,
        "hit_at_3": hit_at_k(sources, expected_sources),
        "reciprocal_rank": reciprocal_rank(sources, expected_sources),
        "latency_ms": latency_ms,
    }


def _summary(case_reports: list[dict], version: str) -> dict:
    results = [case[version] for case in case_reports]
    return {
        "hit_rate_at_3": average([result["hit_at_3"] for result in results]),
        "mrr_at_3": average([result["reciprocal_rank"] for result in results]),
        "average_latency_ms": average([result["latency_ms"] for result in results]),
    }


if __name__ == "__main__":
    report = run_evaluation()
    print(json.dumps({
        "dataset_size": report["dataset_size"],
        "baseline": report["baseline"],
        "rerank": report["rerank"],
        "report_path": str(REPORT_PATH),
    }, ensure_ascii=False, indent=2))
