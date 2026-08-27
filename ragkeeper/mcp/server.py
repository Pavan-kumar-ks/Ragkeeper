from mcp.server import MCPServer

from .. import state
from ..config import get_settings
from ..retrieval import BM25Index, HybridRetriever, Reranker, RetrievalPipeline
from ..vectorstore import get_client, get_embeddings, get_vectorstore

_MAX_TOP_K = 20

mcp = MCPServer("ragkeeper")

_settings = get_settings()
_client = get_client(_settings.qdrant_url)
_embeddings = get_embeddings(_settings.embedding_model)
_vectorstore = get_vectorstore(_client, _settings.qdrant_collection, _embeddings)
_bm25_index = BM25Index(_client, _settings.qdrant_collection)
_hybrid_retriever = HybridRetriever(_vectorstore, _bm25_index, k=_settings.top_k, candidate_k=20)
_reranker = Reranker()


def _pipeline(top_k: int) -> RetrievalPipeline:
    return RetrievalPipeline(_hybrid_retriever, _reranker, k=top_k, pool_size=max(top_k * 5, 20))


@mcp.tool()
def search_docs(query: str, top_k: int = 4) -> list[dict]:
    """Search the indexed LangChain docs via RAGKeeper's hybrid (dense + BM25) retrieval and
    cross-encoder reranking pipeline. Returns ranked chunks with source metadata; does not
    call an LLM."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    k = max(1, min(top_k, _MAX_TOP_K))

    ranked = _pipeline(k).retrieve_with_score(query)
    return [
        {
            "source_path": doc.metadata.get("source_path"),
            "section_title": doc.metadata.get("section_title"),
            "header_hierarchy": doc.metadata.get("header_hierarchy"),
            "content": doc.metadata.get("raw_content", doc.page_content),
            "rerank_score": score,
            "content_hash": doc.metadata.get("content_hash"),
            "commit_hash": doc.metadata.get("commit_hash"),
        }
        for doc, score in ranked
    ]


@mcp.tool()
def get_index_health() -> dict:
    """Report RAGKeeper index health: Qdrant point count, embedding model, latest sync-run
    summary from the local state DB, and whether state/vector store appear consistent."""
    conn = state.init_db(_settings.state_db_path)
    try:
        latest = state.get_latest_sync_run(conn)
    finally:
        conn.close()

    collection_exists = _client.collection_exists(_settings.qdrant_collection)
    point_count = _client.count(_settings.qdrant_collection, exact=True).count if collection_exists else 0

    notes = []
    if not collection_exists:
        notes.append("Qdrant collection does not exist")
    if latest is None:
        notes.append("no sync runs recorded yet")
    elif latest["status"] == "success" and point_count == 0:
        notes.append("last sync reported success but collection is empty")
    elif latest["status"] != "success":
        notes.append(f"last sync run did not succeed: {latest.get('error')}")

    return {
        "collection": _settings.qdrant_collection,
        "collection_exists": collection_exists,
        "point_count": point_count,
        "embedding_model": _settings.embedding_model,
        "last_sync": latest,
        "consistent": not notes,
        "notes": notes,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
