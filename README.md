# RAGKeeper — Phase 1 (Baseline RAG)

A RAG chatbot over the LangChain documentation. This is the baseline pipeline: clone docs,
chunk by Markdown headers, embed (locally, via sentence-transformers), store in Qdrant,
retrieve + generate (via Groq) with citations.

Freshness-aware incremental re-sync (Phase 2) is not implemented yet — the chunk metadata
schema (`content_hash`, `file_content_hash`, `commit_hash`) is already in place to support it.

## Setup

1. Copy `.env.example` to `.env` and fill in `GROQ_API_KEY`. Embeddings run locally
   (`sentence-transformers/all-MiniLM-L6-v2`), so no embedding API key is needed.
2. Install dependencies (into your existing `venv`):
   ```
   pip install -r requirements.txt
   ```
3. Start Qdrant:
   ```
   docker compose up -d
   ```
   Dashboard: http://localhost:6333/dashboard

## Usage

Ingest (clone LangChain repo + chunk + embed + upsert into Qdrant):
```
python main.py ingest
```
Use `--recreate` to drop and rebuild the collection from scratch.

Chat:
```
python main.py chat
```
Ask questions like "How do I create an agent in LangChain?" — answers are grounded in the
retrieved doc chunks, with source file paths printed below each answer.

## Verification

- Check the ingestion log for the discovered file count and final Qdrant point count.
- Re-run `python main.py ingest` and confirm the point count doesn't grow (idempotent upsert).
- Inspect payloads at http://localhost:6333/dashboard to confirm metadata fields are populated.
