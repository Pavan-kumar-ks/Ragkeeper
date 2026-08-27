import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from ragkeeper import state
from ragkeeper.rag_chain import RagKeeperChat

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the follow-up question as a single standalone search query, resolving "
            "pronouns/references using the conversation history. Output only the query, "
            "nothing else.",
        ),
        ("human", "History:\n{history}\n\nFollow-up: {question}"),
    ]
)


def _condense(llm, history: list[dict], question: str) -> str:
    if not history:
        return question
    hist_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    try:
        resp = llm.invoke(_CONDENSE_PROMPT.format_messages(history=hist_text, question=question))
        return resp.content.strip() or question
    except Exception:
        return question


def _save_feedback(settings, query_id: int, feedback: str) -> None:
    conn = state.init_db(settings.state_db_path)
    try:
        state.update_feedback(conn, query_id, feedback)
    finally:
        conn.close()
    st.toast(f"Feedback recorded: {feedback}")


def _render_assistant(msg: dict, settings) -> None:
    st.write(msg["content"])

    if msg.get("sources"):
        with st.expander(f"Sources ({len(msg['sources'])})"):
            for src in msg["sources"]:
                st.markdown(f"- `{src}`")

    if msg.get("trace"):
        with st.expander("Retrieval details"):
            st.dataframe(
                [
                    {
                        "rank": t["final_rank"],
                        "source": t["source_path"],
                        "section": t["section_title"],
                        "dense_rank": t["dense_rank"],
                        "bm25_rank": t["bm25_rank"],
                        "rrf_score": t["rrf_score"],
                        "rerank_score": t["rerank_score"],
                    }
                    for t in msg["trace"]
                ],
                use_container_width=True,
            )

    lat = msg.get("latency") or {}
    if lat:
        cost = msg.get("cost_usd")
        cost_str = "N/A" if cost is None else f"${cost:.5f}"
        st.caption(
            f"retrieval {lat.get('retrieval_ms', 0):.0f}ms · "
            f"generation {lat.get('generation_ms', 0):.0f}ms · "
            f"tokens {msg.get('prompt_tokens', 0)}+{msg.get('completion_tokens', 0)} · "
            f"cost {cost_str}"
        )

    if msg.get("query_id") is not None:
        up, down = st.columns(2, gap="small")
        if up.button("👍", key=f"up-{msg['query_id']}", use_container_width=True):
            _save_feedback(settings, msg["query_id"], "up")
        if down.button("👎", key=f"down-{msg['query_id']}", use_container_width=True):
            _save_feedback(settings, msg["query_id"], "down")


def render(chat: RagKeeperChat, settings) -> None:
    header, clear = st.columns([6, 1])
    header.subheader("Ask RAGKeeper")
    if clear.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_box = st.container()
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    _render_assistant(msg, settings)
                else:
                    st.write(msg["content"])

    question = st.chat_input("Ask about LangChain...")
    if not question:
        return

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Retrieving + generating..."):
        retrieval_query = _condense(chat.llm, history, question)
        result = chat.answer(question, retrieval_query=retrieval_query)

        conn = state.init_db(settings.state_db_path)
        try:
            query_id = state.log_query(
                conn,
                question,
                retrieval_query,
                result["sources"],
                result["retrieval_latency_ms"],
                result["generation_latency_ms"],
                result["prompt_tokens"],
                result["completion_tokens"],
                result["cost_usd"],
            )
        finally:
            conn.close()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "trace": result["trace"],
            "latency": {
                "retrieval_ms": result["retrieval_latency_ms"],
                "generation_ms": result["generation_latency_ms"],
            },
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "cost_usd": result["cost_usd"],
            "query_id": query_id,
        }
    )
    st.rerun()
