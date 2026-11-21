"""Export top-k retrieval results from the supplied Phase 1 gold CSV."""

import argparse
import csv
import os
from pathlib import Path
from time import perf_counter

from papertrail.cli import DEFAULT_MODEL
from papertrail.embeddings import MiniLM
from papertrail.library import Library


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("library", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    library = Library(args.library, MiniLM(os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)))
    rows = []
    with args.gold.open(encoding="utf-8-sig", newline="") as source:
        for question in csv.DictReader(source):
            if question["target_document"] == "NONE":
                continue
            started = perf_counter()
            hits = library.search(question["question"], args.top_k)
            elapsed_ms = (perf_counter() - started) * 1000
            for rank, hit in enumerate(hits, start=1):
                rows.append({
                    "question_id": question["id"],
                    "question": question["question"],
                    "expected_answer": question["expected_answer"],
                    "rank": rank,
                    "paper_id": hit["paper_id"],
                    "page_start": hit["page_number"],
                    "page_end": hit["page_number_end"],
                    "chunk_id": hit["chunk_id"],
                    "score": hit["score"],
                    "backend": hit["ingestion_backend"],
                    "headings": " > ".join(hit["headings"]),
                    "text": hit["text"],
                    "query_elapsed_ms": elapsed_ms,
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
