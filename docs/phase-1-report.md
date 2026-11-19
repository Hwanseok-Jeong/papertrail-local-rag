# Phase 1 · Local PDF retrieval

## Delivered

- `ingest` copies text PDFs into a SHA-256-addressed local library, skips pages without extractable text, and stores page-scoped passages in SQLite.
- `search` ranks normalized CPU embeddings using FAISS. Results include stable chunk IDs, similarity scores, titles, local PDF paths, and original PDF page numbers.
- `rebuild-index` recreates FAISS from SQLite. Retrieval also repairs a missing, unreadable, or out-of-sync index.
- The README includes setup, real-PDF smoke commands, and the human-labeling protocol. The evaluator computes page-level retrieval metrics from JSONL labels.

## Checks run

| Command | Result |
| --- | --- |
| `.\.venv\Scripts\python.exe -m compileall -q src tests evaluation` | Passed |
| `.\.venv\Scripts\python.exe -m pytest -q` | 4 passed on Windows, Python 3.13.13 |
| `git check-ignore .env data/test.pdf evaluation/questions.jsonl example.pdf` | All listed paths ignored |

Automated tests generate their own PDF corpus and inject a deterministic encoder. They cover page association, token bounds, empty PDFs, duplicate ingestion, ranking, fresh-process reload, and recovery after index corruption or a SQLite rollback.

## Acceptance gate

- The full MiniLM CLI smoke test is pending. The automated suite uses a fixture encoder.

- **PENDING USER VALIDATION** · Ingest at least two permitted research PDFs, repeat the search in a fresh process, verify cited pages, and re-ingest to confirm duplicate detection.
- **PENDING USER VALIDATION** · Label 20–30 questions before tuning, reserve a holdout, and record Recall@1/5, MRR@5, page-reference accuracy, latency, and misses. No human-labeled retrieval scores are available.
- Phase 2 requires explicit approval after Phase 1 review.

## Limitations

The CLI handles English text PDFs. Scanned pages require OCR outside Phase 1. Author metadata depends on the PDF, and DOI/year fields are reserved for later enrichment. An interruption can leave an unreferenced content-addressed PDF file, which can be reused if the document is ingested again.
