"""Command-line entry point for local PDF retrieval."""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from .embeddings import MiniLM
from .library import Library
from .rag import ask

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    parser = argparse.ArgumentParser(prog="papertrail")
    parser.add_argument("--data-dir", type=Path, default=os.getenv("DATA_DIR", "./data"))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("folder", type=Path)
    ingest.add_argument("--chunk-tokens", type=int, default=os.getenv("CHUNK_TOKENS", "180"))
    ingest.add_argument("--overlap", type=int, default=os.getenv("CHUNK_OVERLAP_TOKENS", "30"))
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=os.getenv("TOP_K", "5"))
    commands.add_parser("rebuild-index")
    answer = commands.add_parser("ask")
    answer.add_argument("query")
    answer.add_argument("--top-k", type=int, default=os.getenv("TOP_K", "5"))
    args = parser.parse_args()

    paths = []
    if args.command == "ingest":
        if not args.folder.is_dir():
            parser.error("ingest requires an existing folder")
        paths = sorted(p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
        if not paths:
            parser.error("no PDFs found in the folder")
    if args.command in {"search", "ask"} and (not args.query.strip() or args.top_k < 1):
        parser.error("search requires a nonblank query and positive top-k")

    try:
        library = Library(args.data_dir, MiniLM(args.model))
        if args.command in {"search", "ask"}:
            hits = library.search(args.query, args.top_k)
            print(json.dumps(ask(args.query, hits) if args.command == "ask" else hits, indent=2))
        elif args.command == "rebuild-index":
            print(json.dumps({"indexed_chunks": library.rebuild()}))
        else:
            library.length = args.chunk_tokens
            library.overlap = args.overlap
            failed = False
            for path in paths:
                try:
                    print(json.dumps(library.ingest(path)))
                except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
                    print(f"{path.name}: {error}", file=sys.stderr)
                    failed = True
            return int(failed)
    except (ImportError, OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        print(f"PaperTrail: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
