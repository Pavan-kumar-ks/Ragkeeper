import json
from pathlib import Path

import streamlit as st

RESULTS_DIR = Path("eval_results")


def render() -> None:
    st.subheader("Evaluation Runs")
    files = sorted(RESULTS_DIR.glob("eval_*.json"))
    if not files:
        st.info("No eval reports found. Run `python main.py eval`.")
        return

    runs = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        agg = data["aggregate"]
        runs.append(
            {
                "run": f.stem,
                "precision@k": agg["retrieval"]["precision_at_k"],
                "recall@k": agg["retrieval"]["recall_at_k"],
                "mrr": agg["retrieval"]["mrr"],
                "ndcg@k": agg["retrieval"]["ndcg_at_k"],
                "faithfulness": agg["generation"]["faithfulness"],
                "answer_relevance": agg["generation"]["answer_relevance"],
                "correctness": agg["generation"]["correctness"],
                "avg_latency_ms": agg["end_to_end"]["avg_latency_ms"],
                "total_cost_usd": agg["end_to_end"]["total_cost_usd"],
            }
        )
    st.dataframe(runs, use_container_width=True)
    st.line_chart({r["run"]: r["correctness"] for r in runs})

    labels = [r["run"] for r in runs]
    chosen = st.selectbox("Inspect run", labels, index=len(labels) - 1)
    data = json.loads((RESULTS_DIR / f"{chosen}.json").read_text(encoding="utf-8"))

    st.markdown("**Per-question results**")
    rows = [
        {
            "question": r["question"],
            "correctness": r["generation"]["correctness"],
            "faithfulness": r["generation"]["faithfulness"],
            "answer_relevance": r["generation"]["answer_relevance"],
            "precision@k": r["retrieval"]["precision_at_k"],
            "recall@k": r["retrieval"]["recall_at_k"],
            "retrieved_sources": ", ".join(r["retrieved_sources"]),
        }
        for r in data["results"]
    ]
    st.dataframe(rows, use_container_width=True)

    question_labels = [r["question"] for r in data["results"]]
    picked = st.selectbox("Drill into a question", question_labels)
    detail = next(r for r in data["results"] if r["question"] == picked)
    st.json(detail)
