import datetime as dt
import time

from langchain_core.documents import Document
from tqdm import tqdm

from . import state
from .config import get_settings
from .git_sync import ensure_repo_synced
from .hashing import deterministic_point_id, sha256_hex
from .loader import discover_doc_files, read_file
from .splitter import header_hierarchy, split_markdown
from .vectorstore import (
    EMBEDDING_DIM,
    delete_by_source_path,
    ensure_collection,
    get_client,
    get_embeddings,
    get_vectorstore,
)


def _embed_file(client, vectorstore, settings, path, raw_text, rel_path, file_content_hash, commit_hash, ingested_at) -> int:
    chunks = split_markdown(raw_text, settings.chunk_size, settings.chunk_overlap)
    delete_by_source_path(client, settings.qdrant_collection, rel_path)
    if not chunks:
        return 0

    docs: list[Document] = []
    ids: list[str] = []
    for idx, chunk in enumerate(chunks):
        hierarchy = header_hierarchy(chunk.metadata)
        metadata = {
            "source_path": rel_path,
            "header_hierarchy": hierarchy,
            "section_title": hierarchy[-1] if hierarchy else path.stem,
            "chunk_index": idx,
            "content_hash": sha256_hex(chunk.page_content),
            "file_content_hash": file_content_hash,
            "commit_hash": commit_hash,
            "repo_url": settings.langchain_repo_url,
            "file_ext": path.suffix,
            "ingested_at": ingested_at,
        }
        docs.append(Document(page_content=chunk.page_content, metadata=metadata))
        ids.append(deterministic_point_id(rel_path, idx))

    vectorstore.add_documents(docs, ids=ids)
    return len(docs)


def run_ingestion(recreate: bool = False) -> None:
    settings = get_settings()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    start_time = time.perf_counter()

    conn = state.init_db(settings.state_db_path)

    files_added = files_updated = files_deleted = files_unchanged = 0
    chunks_added = chunks_deleted = 0
    commit_hash = None

    try:
        print(f"Syncing LangChain repo at {settings.langchain_repo_path} ...")
        commit_hash = ensure_repo_synced(settings.langchain_repo_url, settings.langchain_repo_path)
        print(f"Repo synced at commit {commit_hash}")

        files = discover_doc_files(settings.langchain_repo_path, settings.docs_root_subdir)
        print(f"Discovered {len(files)} markdown/mdx files under {settings.docs_root_subdir}/")

        client = get_client(settings.qdrant_url)
        ensure_collection(client, settings.qdrant_collection, vector_size=EMBEDDING_DIM, recreate=recreate)
        if recreate:
            state.reset(conn)

        embeddings = get_embeddings(settings.embedding_model)
        vectorstore = get_vectorstore(client, settings.qdrant_collection, embeddings)

        previous = state.load_file_state(conn)
        ingested_at = dt.datetime.now(dt.timezone.utc).isoformat()
        discovered_rel_paths: set[str] = set()

        for path in tqdm(files, desc="Ingesting docs"):
            raw_text = read_file(path)
            if not raw_text.strip():
                continue

            rel_path = str(path.relative_to(settings.langchain_repo_path)).replace("\\", "/")
            discovered_rel_paths.add(rel_path)
            file_content_hash = sha256_hex(raw_text)

            prior = previous.get(rel_path)
            if prior is None:
                category = "added"
            elif prior["file_content_hash"] != file_content_hash:
                category = "updated"
            else:
                category = "unchanged"

            if category == "unchanged":
                files_unchanged += 1
                continue

            chunk_count = _embed_file(
                client, vectorstore, settings, path, raw_text, rel_path,
                file_content_hash, commit_hash, ingested_at,
            )
            state.upsert_file_state(conn, rel_path, file_content_hash, commit_hash, chunk_count, ingested_at)
            chunks_added += chunk_count

            if category == "added":
                files_added += 1
            else:
                files_updated += 1

        deleted_paths = set(previous.keys()) - discovered_rel_paths
        for rel_path in deleted_paths:
            delete_by_source_path(client, settings.qdrant_collection, rel_path)
            chunks_deleted += previous[rel_path]["chunk_count"]
            state.delete_file_state(conn, rel_path)
            files_deleted += 1

        count = client.count(settings.qdrant_collection, exact=True).count
        duration_s = time.perf_counter() - start_time
        state.record_sync_run(
            conn, started_at, dt.datetime.now(dt.timezone.utc).isoformat(), commit_hash,
            files_added, files_updated, files_deleted, files_unchanged,
            chunks_added, chunks_deleted, duration_s, status="success",
        )

        print(
            f"\nSync complete in {duration_s:.1f}s — "
            f"added: {files_added}, updated: {files_updated}, deleted: {files_deleted}, unchanged: {files_unchanged}"
        )
        print(f"Chunks embedded: {chunks_added}, chunks removed: {chunks_deleted}. Collection now has {count} points.")
    except Exception as exc:
        duration_s = time.perf_counter() - start_time
        state.record_sync_run(
            conn, started_at, dt.datetime.now(dt.timezone.utc).isoformat(), commit_hash,
            files_added, files_updated, files_deleted, files_unchanged,
            chunks_added, chunks_deleted, duration_s, status="error", error=str(exc),
        )
        raise
    finally:
        conn.close()
