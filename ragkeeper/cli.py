import argparse

from .eval.run_eval import run_evaluation
from .ingest import run_ingestion
from .rag_chain import RagKeeperChat


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

        answer, sources = chat.answer(question)
        print(f"\nRAGKeeper: {answer}\n")
        if sources:
            print("Sources:")
            for src in sources:
                print(f"  - {src}")
        print()


def cmd_eval(args: argparse.Namespace) -> None:
    run_evaluation(k=args.k)


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
