import argparse

from . import state
from .config import get_settings
from .eval.run_eval import run_evaluation
from .ingest import run_ingestion
from .rag_chain import RagKeeperChat
from .vectorstore import get_client


def cmd_ingest(args: argparse.Namespace) -> None:
    run_ingestion(recreate=args.recreate)


def cmd_chat(_args: argparse.Namespace) -> None:
    chat = RagKeeperChat()
    print("RAGKeeper chat. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = chat.answer(question)
        print(f"\nRAGKeeper: {result['answer']}\n")
        if result["sources"]:
            print("Sources:")
            for src in result["sources"]:
                print(f"  - {src}")
        print()


def cmd_eval(args: argparse.Namespace) -> None:
    run_evaluation(k=args.k)


def cmd_mcp(_args: argparse.Namespace) -> None:
    from .mcp.server import main as run_mcp_server

    run_mcp_server()


def cmd_schedule(args: argparse.Namespace) -> None:
    from .scheduler import run_scheduler

    run_scheduler(interval_hours=args.interval_hours)


def cmd_dashboard(_args: argparse.Namespace) -> None:
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)


def cmd_status(_args: argparse.Namespace) -> None:
    settings = get_settings()
    conn = state.init_db(settings.state_db_path)
    try:
        latest = state.get_latest_sync_run(conn)
    finally:
        conn.close()

    client = get_client(settings.qdrant_url)
    if client.collection_exists(settings.qdrant_collection):
        point_count = client.count(settings.qdrant_collection, exact=True).count
    else:
        point_count = 0

    print(f"Collection: {settings.qdrant_collection} ({point_count} points)\n")

    if latest is None:
        print("No sync runs recorded yet. Run 'python main.py ingest' first.")
        return

    print(f"Last sync run: {latest['status']} at {latest['finished_at']}")
    print(f"Commit: {latest['commit_hash']}")
    print(
        f"Files — added: {latest['files_added']}, updated: {latest['files_updated']}, "
        f"deleted: {latest['files_deleted']}, unchanged: {latest['files_unchanged']}"
    )
    print(f"Chunks — added: {latest['chunks_added']}, deleted: {latest['chunks_deleted']}")
    print(f"Duration: {latest['duration_s']:.1f}s")
    if latest["error"]:
        print(f"Error: {latest['error']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragkeeper", description="RAGKeeper: freshness-aware RAG for LangChain docs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Sync LangChain repo docs and (re)index them in Qdrant")
    ingest_parser.add_argument("--recreate", action="store_true", help="Drop and recreate the Qdrant collection before ingesting")
    ingest_parser.set_defaults(func=cmd_ingest)

    chat_parser = subparsers.add_parser("chat", help="Interactive Q&A over the indexed LangChain docs")
    chat_parser.set_defaults(func=cmd_chat)

    eval_parser = subparsers.add_parser("eval", help="Run the retrieval/generation/end-to-end eval harness against the golden set")
    eval_parser.add_argument("--k", type=int, default=None, help="Override retrieval top-k (defaults to settings.top_k)")
    eval_parser.set_defaults(func=cmd_eval)

    status_parser = subparsers.add_parser("status", help="Show index health: last sync run + collection point count")
    status_parser.set_defaults(func=cmd_status)

    mcp_parser = subparsers.add_parser("mcp", help="Run the MCP server (search_docs, get_index_health) over stdio")
    mcp_parser.set_defaults(func=cmd_mcp)

    dashboard_parser = subparsers.add_parser("dashboard", help="Launch the Streamlit dashboard")
    dashboard_parser.set_defaults(func=cmd_dashboard)

    schedule_parser = subparsers.add_parser(
        "schedule", help="Run ingest on a repeating interval (freshness-aware auto-sync)"
    )
    schedule_parser.add_argument(
        "--interval-hours", type=float, default=24.0, help="Hours between sync runs (default: 24)"
    )
    schedule_parser.set_defaults(func=cmd_schedule)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
