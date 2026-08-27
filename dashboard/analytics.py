import json

import streamlit as st

from ragkeeper import state


def render(settings) -> None:
    st.subheader("Query Analytics")
    conn = state.init_db(settings.state_db_path)
    try:
        rows = state.get_recent_queries(conn, limit=100)
    finally:
        conn.close()

    if not rows:
        st.info("No queries logged yet — ask something in the Chat tab.")
        return

    display_rows = [
        {
            "id": r["id"],
            "asked_at": r["asked_at"],
            "question": r["question"],
            "retrieval_query": r["retrieval_query"],
            "sources": ", ".join(json.loads(r["sources"]) if r["sources"] else []),
            "retrieval_ms": r["retrieval_latency_ms"],
            "generation_ms": r["generation_latency_ms"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "cost_usd": r["cost_usd"],
            "feedback": r["feedback"],
        }
        for r in rows
    ]
    st.dataframe(display_rows, use_container_width=True)

    total_cost = sum(r["cost_usd"] for r in rows if r["cost_usd"] is not None)
    avg_latency = sum(
        (r["retrieval_latency_ms"] or 0) + (r["generation_latency_ms"] or 0) for r in rows
    ) / len(rows)
    cols = st.columns(3)
    cols[0].metric("Logged queries", len(rows))
    cols[1].metric("Avg latency (ms)", f"{avg_latency:.0f}")
    cols[2].metric("Total cost (USD)", f"{total_cost:.5f}")
