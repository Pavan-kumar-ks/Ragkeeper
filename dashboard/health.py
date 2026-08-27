import streamlit as st

from ragkeeper import state
from ragkeeper.health import compute_index_health


def render(settings, client) -> None:
    st.subheader("Index Health")
    health = compute_index_health(settings, client)

    cols = st.columns(4)
    cols[0].metric("Qdrant points", health["point_count"])
    cols[1].metric("Embedding model", health["embedding_model"])
    cols[2].metric("Collection", health["collection"])
    cols[3].metric("Consistent", "OK" if health["consistent"] else "ISSUE")

    if health["notes"]:
        st.warning("\n".join(health["notes"]))

    if health["last_sync"]:
        st.markdown("**Last sync run**")
        st.json(health["last_sync"])

    st.subheader("Sync History")
    conn = state.init_db(settings.state_db_path)
    try:
        history = state.get_sync_run_history(conn, limit=20)
    finally:
        conn.close()

    if history:
        st.dataframe(history, use_container_width=True)
    else:
        st.info("No sync runs recorded yet.")


def render_system_health(settings, client) -> None:
    st.subheader("System Health")
    checks = []

    try:
        client.collection_exists(settings.qdrant_collection)
        checks.append(("Qdrant", "OK"))
    except Exception as exc:
        checks.append(("Qdrant", f"DOWN ({exc})"))

    try:
        conn = state.init_db(settings.state_db_path)
        conn.close()
        checks.append(("SQLite state DB", "OK"))
    except Exception as exc:
        checks.append(("SQLite state DB", f"DOWN ({exc})"))

    checks.append(("Embedding model", settings.embedding_model))
    checks.append(("Reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    checks.append(("Query expansion", "enabled" if settings.enable_query_expansion else "disabled"))

    st.table({"component": [c[0] for c in checks], "status": [c[1] for c in checks]})
