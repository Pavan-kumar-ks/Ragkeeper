import math


def _is_relevant(source_path: str, expected: list[str]) -> bool:
    return any(source_path.endswith(exp) for exp in expected)


def precision_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for src in top_k if _is_relevant(src, expected))
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for src in top_k if _is_relevant(src, expected))
    return min(hits, len(expected)) / len(expected)


def mrr(retrieved: list[str], expected: list[str]) -> float:
    for rank, src in enumerate(retrieved, start=1):
        if _is_relevant(src, expected):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    top_k = retrieved[:k]
    dcg = sum(
        (1.0 if _is_relevant(src, expected) else 0.0) / math.log2(i + 1)
        for i, src in enumerate(top_k, start=1)
    )
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
