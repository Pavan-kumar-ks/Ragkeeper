import datetime as dt
import json
import time
from pathlib import Path

import groq
from langchain_groq import ChatGroq

from ..config import get_settings
from ..rag_chain import PROMPT, format_context
from ..retrieval import BM25Index, HybridRetriever, Reranker, RetrievalPipeline
from ..vectorstore import get_client, get_embeddings, get_vectorstore
from .golden_set import GOLDEN_SET
from .judge import build_judge, judge_answer
from .pricing import estimate_cost
from .retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

RESULTS_DIR = Path("eval_results")


def _invoke_with_retry(fn, *args, max_retries: int = 6, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except groq.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait_s = 2 * (attempt + 1)
            print(f"  Rate limited, retrying in {wait_s}s...")
            time.sleep(wait_s)


def _evaluate_one(retriever, llm, judge_llm, model: str, k: int, entry: dict) -> dict:
    question = entry["question"]
    expected_sources = entry["expected_sources"]
    reference_answer = entry["reference_answer"]

    retrieval_start = time.perf_counter()
    docs = retriever.retrieve(question)
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

    retrieved_sources = [doc.metadata.get("source_path", "") for doc in docs]

    generation_start = time.perf_counter()
    context = format_context(docs)
    messages = PROMPT.format_messages(context=context, question=question)
    ai_message = _invoke_with_retry(llm.invoke, messages)
    generation_latency_ms = (time.perf_counter() - generation_start) * 1000

    answer_text = ai_message.content
    token_usage = ai_message.response_metadata.get("token_usage", {})
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)

    scores = _invoke_with_retry(judge_answer, judge_llm, question, context, answer_text, reference_answer)

    return {
        "question": question,
        "answer": answer_text,
        "retrieved_sources": retrieved_sources,
        "expected_sources": expected_sources,
        "retrieval": {
            "precision_at_k": precision_at_k(retrieved_sources, expected_sources, k),
            "recall_at_k": recall_at_k(retrieved_sources, expected_sources, k),
            "mrr": mrr(retrieved_sources, expected_sources),
            "ndcg_at_k": ndcg_at_k(retrieved_sources, expected_sources, k),
            "latency_ms": retrieval_latency_ms,
        },
        "generation": {
            "faithfulness": scores.faithfulness,
            "answer_relevance": scores.answer_relevance,
            "correctness": scores.correctness,
            "justification": scores.justification,
        },
        "end_to_end": {
            "latency_ms": retrieval_latency_ms + generation_latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": token_usage.get("total_tokens", prompt_tokens + completion_tokens),
            "groq_completion_time_s": token_usage.get("completion_time"),
            "groq_prompt_time_s": token_usage.get("prompt_time"),
            "cost_usd": estimate_cost(model, prompt_tokens, completion_tokens),
        },
    }


def _aggregate(results: list[dict]) -> dict:
    n = len(results)

    def avg(path: tuple[str, str]) -> float:
        section, key = path
        values = [r[section][key] for r in results if r[section].get(key) is not None]
        return sum(values) / len(values) if values else 0.0

    return {
        "retrieval": {
            "precision_at_k": avg(("retrieval", "precision_at_k")),
            "recall_at_k": avg(("retrieval", "recall_at_k")),
            "mrr": avg(("retrieval", "mrr")),
            "ndcg_at_k": avg(("retrieval", "ndcg_at_k")),
        },
        "generation": {
            "faithfulness": avg(("generation", "faithfulness")),
            "answer_relevance": avg(("generation", "answer_relevance")),
            "correctness": avg(("generation", "correctness")),
        },
        "end_to_end": {
            "avg_latency_ms": avg(("end_to_end", "latency_ms")),
            "total_tokens": sum(r["end_to_end"]["total_tokens"] for r in results),
            "total_cost_usd": sum(
                r["end_to_end"]["cost_usd"] for r in results if r["end_to_end"]["cost_usd"] is not None
            )
            if any(r["end_to_end"]["cost_usd"] is not None for r in results)
            else None,
        },
        "num_questions": n,
    }


def _print_report(aggregate: dict) -> None:
    print("\n=== RAGKeeper Eval Report ===")
    print(f"Questions evaluated: {aggregate['num_questions']}\n")

    print("Retrieval:")
    for key, value in aggregate["retrieval"].items():
        print(f"  {key:<15} {value:.3f}")

    print("\nGeneration (LLM-judge, 1-5):")
    for key, value in aggregate["generation"].items():
        print(f"  {key:<15} {value:.2f}")

    print("\nEnd-to-end:")
    e2e = aggregate["end_to_end"]
    print(f"  avg_latency_ms  {e2e['avg_latency_ms']:.0f}")
    print(f"  total_tokens    {e2e['total_tokens']}")
    cost = e2e["total_cost_usd"]
    print(f"  total_cost_usd  {'N/A (unpriced model)' if cost is None else f'{cost:.5f}'}")


def run_evaluation(k: int | None = None) -> None:
    settings = get_settings()
    eval_k = k or settings.top_k

    client = get_client(settings.qdrant_url)
    embeddings = get_embeddings(settings.embedding_model)
    vectorstore = get_vectorstore(client, settings.qdrant_collection, embeddings)

    print("Building BM25 index...")
    bm25_index = BM25Index(client, settings.qdrant_collection)
    hybrid_retriever = HybridRetriever(vectorstore, bm25_index, k=eval_k, candidate_k=20)
    reranker = Reranker()
    retriever = RetrievalPipeline(hybrid_retriever, reranker, k=eval_k, pool_size=20)

    llm = ChatGroq(model=settings.groq_model, temperature=0, api_key=settings.groq_api_key)
    judge_llm = build_judge(settings.groq_model, settings.groq_api_key)

    results = []
    for entry in GOLDEN_SET:
        print(f"Evaluating: {entry['question']}")
        results.append(_evaluate_one(retriever, llm, judge_llm, settings.groq_model, eval_k, entry))

    aggregate = _aggregate(results)
    _print_report(aggregate)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = RESULTS_DIR / f"eval_{timestamp}.json"
    report_path.write_text(
        json.dumps({"aggregate": aggregate, "results": results}, indent=2), encoding="utf-8"
    )
    print(f"\nFull report written to {report_path}")
