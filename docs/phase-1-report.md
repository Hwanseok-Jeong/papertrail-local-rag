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

## MiniLM smoke test

On Windows with sentence-transformers 6.3.0, CPU MiniLM indexed two locally generated one-page PDFs. The first command downloaded the model to `data/model-cache`. The subsequent commands used the same cache with `HF_HUB_OFFLINE=1`.

| Command | Result |
| --- | --- |
| `.\.venv\Scripts\papertrail.exe --data-dir data/smoke-library ingest data/smoke-input` | Two papers, one chunk each |
| `.\.venv\Scripts\papertrail.exe --data-dir data/smoke-library search "Where did the orchid grow?"` | Correct page ranked first at 0.742 cosine similarity |
| Repeat the ingest command | Both PDFs reported as duplicates |
| `.\.venv\Scripts\papertrail.exe --data-dir data/smoke-library rebuild-index` | Two indexed chunks |
| `.\.venv\Scripts\python.exe evaluation/evaluate.py data/smoke-questions.jsonl --data-dir data/smoke-library --split development` | Evaluator completed on one answerable and one unanswerable synthetic query |

For the single answerable query, Recall@1, Recall@5, and MRR@5 were all 1.0. All four returned page references matched the source excerpts. Mean query time was 30.0 ms across the two queries, after model initialization. The unanswerable query's top cosine score was 0.241. These figures measure the synthetic smoke fixture only.

## Acceptance gate

- **REAL-PDF BASELINE RECORDED LOCALLY** · Two user-selected PDFs and the supplied gold CSV were evaluated. All 18 ranked lists reproduced after restart, both reimports were duplicates, and all 90 excerpts matched their cited PDF pages. Evidence-based Hit@1 was 2/18, Hit@5 was 7/18, and MRR@5 was 0.2056. Four unanswerable prompts were excluded. See the local report and inspection files under `data/phase1-validation/`.
- **PENDING HOLDOUT VALIDATION** · The guide designates the supplied questions as a development set. Measure any retrieval changes on a separate holdout before drawing general conclusions.
- Phase 2 requires explicit approval after Phase 1 review.

## Limitations

The CLI handles English text PDFs. Scanned pages require OCR outside Phase 1. Author metadata depends on the PDF, and DOI/year fields are reserved for later enrichment. An interruption can leave an unreferenced content-addressed PDF file, which can be reused if the document is ingested again.
