import datetime as dt

from langchain_core.documents import Document
from tqdm import tqdm

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


def run_ingestion(recreate: bool = False) -> None:
    settings = get_settings()

    print(f"Syncing LangChain repo at {settings.langchain_repo_path} ...")
    commit_hash = ensure_repo_synced(settings.langchain_repo_url, settings.langchain_repo_path)
    print(f"Repo synced at commit {commit_hash}")

    files = discover_doc_files(settings.langchain_repo_path, settings.docs_root_subdir)
    print(f"Discovered {len(files)} markdown/mdx files under {settings.docs_root_subdir}/")

    client = get_client(settings.qdrant_url)
    ensure_collection(client, settings.qdrant_collection, vector_size=EMBEDDING_DIM, recreate=recreate)

    embeddings = get_embeddings(settings.embedding_model)
    vectorstore = get_vectorstore(client, settings.qdrant_collection, embeddings)

    ingested_at = dt.datetime.now(dt.timezone.utc).isoformat()
    total_chunks = 0

    for path in tqdm(files, desc="Ingesting docs"):
        raw_text = read_file(path)
        if not raw_text.strip():
            continue

        rel_path = str(path.relative_to(settings.langchain_repo_path)).replace("\\", "/")
        file_content_hash = sha256_hex(raw_text)

        chunks = split_markdown(raw_text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            continue

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

        delete_by_source_path(client, settings.qdrant_collection, rel_path)
        vectorstore.add_documents(docs, ids=ids)
        total_chunks += len(docs)

    count = client.count(settings.qdrant_collection, exact=True).count
    print(f"Ingestion complete. {total_chunks} chunks embedded this run. Collection now has {count} points.")
