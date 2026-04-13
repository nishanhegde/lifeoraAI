"""
LifeoraAI RAG — CLI entry point.

Install the package first (from the lifeoraAI/ directory):
    pip install -e .

Then use:
    python main.py ingest          # embed all docs in data/raw/
    python main.py ask             # interactive Q&A loop
    python main.py ask --sources   # show source chunks with each answer
    python main.py ask --debug     # verbose logging
"""
import argparse
import sys

from dotenv import load_dotenv

from core.logging_config import setup_logging
from core.exceptions import LifeoraError, ValidationError
from rag.pipeline import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeoraai",
        description="LifeoraAI — health & wellness RAG assistant",
    )
    parser.add_argument(
        "command",
        choices=["ingest", "ask"],
        help="'ingest' to embed documents, 'ask' for interactive Q&A",
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help="Show source chunks alongside each answer",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    return parser


def run_ingest(pipeline: RAGPipeline) -> None:
    added = pipeline.ingest()
    print(f"Ingestion complete. {added} new chunks added.")


def run_interactive(pipeline: RAGPipeline, show_sources: bool) -> None:
    print("\nLifeoraAI is ready. Ask any health or lifestyle question. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            answer = pipeline.ask(question, show_sources=show_sources)
            print(f"\nLifeoraAI: {answer}\n")
        except ValidationError as exc:
            print(f"\n[Invalid input] {exc}\n")
        except LifeoraError as exc:
            print(f"\n[Error] {exc}\n")


def cli() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    setup_logging(level="DEBUG" if args.debug else "INFO")

    try:
        pipeline = RAGPipeline(config_path=args.config)
    except LifeoraError as exc:
        print(f"[Startup error] {exc}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"[Config not found] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.command == "ingest":
        run_ingest(pipeline)
    elif args.command == "ask":
        run_interactive(pipeline, show_sources=args.sources)


if __name__ == "__main__":
    cli()
