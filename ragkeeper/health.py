from . import state


def compute_index_health(settings, client) -> dict:
    conn = state.init_db(settings.state_db_path)
    try:
        latest = state.get_latest_sync_run(conn)
    finally:
        conn.close()

    collection_exists = client.collection_exists(settings.qdrant_collection)
    point_count = client.count(settings.qdrant_collection, exact=True).count if collection_exists else 0

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
        "collection": settings.qdrant_collection,
        "collection_exists": collection_exists,
        "point_count": point_count,
        "embedding_model": settings.embedding_model,
        "last_sync": latest,
        "consistent": not notes,
        "notes": notes,
    }
