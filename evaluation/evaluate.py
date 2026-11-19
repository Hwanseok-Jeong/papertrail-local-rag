"""Compute page-level retrieval metrics against human-labeled PDF references."""

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from time import perf_counter

import pymupdf

from papertrail.cli import DEFAULT_MODEL
from papertrail.embeddings import MiniLM
from papertrail.library import Library


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=["development", "holdout"], required=True)
    args = parser.parse_args()
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    selected = [question for question in questions if question["split"] == args.split]
    if not selected:
        parser.error("No labeled questions in the selected split")
    for question in selected:
        for reference in question["relevant"]:
            if not re.fullmatch(r"[a-fA-F0-9]{64}", reference["file_sha256"]):
                parser.error(f"{question['id']} has an invalid PDF SHA-256")
            if not isinstance(reference["page_number"], int) or reference["page_number"] < 1:
                parser.error(f"{question['id']} requires one-based PDF page numbers")
    library = Library(args.data_dir, MiniLM(args.model))
    timings, recall_one, recall_five, reciprocal_ranks, failures = [], [], [], [], []
    unanswerable = []
    checked_references = correct_references = 0
    for question in selected:
        relevant = {(ref["file_sha256"].lower(), ref["page_number"]) for ref in question["relevant"]}
        started = perf_counter()
        results = library.search(question["query"], 5)
        timings.append((perf_counter() - started) * 1000)
        pages = [(row["file_sha256"], row["page_number"]) for row in results]
        for row in results:
            with pymupdf.open(library.directory / row["file_path"]) as pdf:
                page_index = row["page_number"] - 1
                checked_references += 1
                if 0 <= page_index < len(pdf):
                    page_text = " ".join(pdf[page_index].get_text(sort=True).split())
                    correct_references += row["text"] in page_text
        if not relevant:
            unanswerable.append({"id": question["id"], "top_score": results[0]["score"] if results else None})
            continue
        recall_one.append(len(set(pages[:1]) & relevant) / len(relevant))
        recall_five.append(len(set(pages) & relevant) / len(relevant))
        rank = next((i for i, page in enumerate(pages, start=1) if page in relevant), None)
        reciprocal_ranks.append(1 / rank if rank else 0)
        if not rank:
            failures.append(question["id"])

    print(json.dumps({
        "model": args.model, "split": args.split, "questions": len(selected),
        "answerable": len(recall_five), "recall_at_1": mean(recall_one) if recall_one else None,
        "recall_at_5": mean(recall_five) if recall_five else None,
        "mrr_at_5": mean(reciprocal_ranks) if reciprocal_ranks else None,
        "mean_query_ms": mean(timings), "no_relevant_page_in_top_5": failures,
        "page_reference_accuracy": correct_references / checked_references if checked_references else None,
        "page_references_checked": checked_references,
        "unanswerable_queries": unanswerable,
    }, indent=2))


if __name__ == "__main__":
    main()
