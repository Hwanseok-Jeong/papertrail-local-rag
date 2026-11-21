# PaperTrail · Local Scientific PDF Retrieval

PaperTrail indexes a folder of scientific PDFs and finds passages by semantic similarity. Search results include a score, paper title, original PDF path, and a one-based PDF page number. Phase 1 runs on CPU and supports English text-based PDFs.

```text
PDF → PyMuPDF pages → tokenizer-aware excerpts → MiniLM → FAISS
                            ↓                            ↓
                     SQLite chunk records ←───── stable chunk IDs
```

## Setup

From the repository root, use Python 3.11 or newer in a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[model,test]'
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux or WSL, use `.venv/bin/python` and `.venv/bin/papertrail`. The embedding model downloads on first use. To store the model in this project's ignored data directory on PowerShell, set `$env:HF_HOME = Join-Path (Get-Location) 'data/model-cache'` before running the CLI. After the model is cached, set `HF_HUB_OFFLINE=1` in your shell to run offline.

## Phase 1 smoke test

Put two or more text-based PDFs you can use in a local folder. Replace the path and query below with your own corpus and a question with a known answer.

```powershell
.\.venv\Scripts\papertrail.exe ingest "C:\path\to\papers"
.\.venv\Scripts\papertrail.exe search "Which normalization method was used?" --top-k 5
.\.venv\Scripts\papertrail.exe search "Which normalization method was used?" --top-k 5
.\.venv\Scripts\papertrail.exe ingest "C:\path\to\papers"
.\.venv\Scripts\papertrail.exe rebuild-index
```

Open the returned PDF under `data/` using its `file_path` and check that the excerpt appears on `page_number`. Check two or three citations. The repeated search uses a new process and should return the same chunk IDs. The second ingest should report `duplicate` for each previously indexed PDF. Rebuild should return the original indexed chunk count.

Use `--data-dir` before a subcommand to choose a separate library. Use `ingest --chunk-tokens` and `--overlap` to adjust the initial 180/30 token settings. Reingesting an unchanged PDF is idempotent. Use a separate data directory to compare chunk settings.

## Retrieval evaluation

Copy `evaluation/questions.example.jsonl` to `evaluation/questions.jsonl`. Before tuning, label about 20–30 questions with all relevant one-based PDF pages using the PDF SHA-256 as the document key. Include unanswerable questions with an empty `relevant` list. Assign a development or holdout split, use development questions to tune, and reserve holdout for the final run.

```powershell
.\.venv\Scripts\python.exe evaluation/evaluate.py evaluation/questions.jsonl --split development
.\.venv\Scripts\python.exe evaluation/evaluate.py evaluation/questions.jsonl --split holdout
```

The evaluator reports page-level Recall@1/5, MRR@5, mean query latency, misses, and whether excerpts match their returned source pages. Unanswerable queries report top scores separately because retrieval ranks available passages even for unrelated questions.

## Current scope

Phase 2 adds local question answering. Install Ollama, start its local server, and select an installed model. For the two-paper development library, use:

```powershell
ollama pull qwen2.5:1.5b
$env:OLLAMA_MODEL = 'qwen2.5:1.5b'
$env:HF_HOME = Join-Path (Get-Location) 'data/model-cache'
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\papertrail.exe --data-dir data/block180/library ask "What were the inclusion criteria for the AIN3 cohort?"
```

For another library, pass its directory to `--data-dir`. Each accepted claim includes its source page and a verbatim evidence quote. Empty evidence, unknown source IDs and invalid quotes yield `insufficient_evidence`. Quote matching verifies attribution. Assess whether the passage supports the claim and answers the entire question during review. Configure the endpoint and request timeout with `OLLAMA_BASE_URL` and `OLLAMA_TIMEOUT` (seconds).

Phase reports and architecture notes:

- [Phase 1 retrieval report](docs/phase-1-report.md)
- [Phase 2 answer-generation report](docs/phase-2-report.md)
- [Development phases and validation gates](docs/development-phases.md)
- [Project specification](PaperTrail_project.md)
