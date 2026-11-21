"""Save local answers and retrieved evidence for claim-by-claim review."""

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

from papertrail.cli import DEFAULT_MODEL
from papertrail.embeddings import MiniLM
from papertrail.library import Library
from papertrail.rag import ask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("library", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    library = Library(args.library, MiniLM(os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.gold.open(encoding="utf-8-sig", newline="") as source, args.output.open(
        "x", encoding="utf-8"
    ) as output:
        for item in csv.DictReader(source):
            start = perf_counter()
            hits = library.search(item["question"], 5)
            answer = ask(item["question"], hits)
            record = dict(id=item["id"], model=os.environ["OLLAMA_MODEL"],
                          question=item["question"], expected=item["expected_answer"],
                          elapsed_seconds=perf_counter() - start, hits=hits, **answer)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
