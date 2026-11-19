# PaperTrail — Scientific Literature Wiki & Local RAG Assistant

## 1. Project objective
Build a **fully local, free, CPU-compatible** scientific literature wiki with retrieval-augmented generation (RAG). Users upload scientific PDFs, browse paper pages and metadata, search passages semantically, and ask questions grounded **only in the uploaded corpus**, with paper/page citations. The app is a portfolio-quality demonstration of Python, retrieval, LLM integration, REST APIs, SQL, testing, and deployment practices.

**Delivery rule:** Every phase is a separately reviewable milestone. Implement -> run automated tests -> run a real-user smoke test -> measure quality -> document results -> request explicit approval before advancing. A passing test suite alone is not sufficient.

**Primary users:** researchers/students reading bioinformatics, genomics, epigenomics, and ML papers. The architecture should remain domain-independent. **Initial scope:** English, text-based PDFs; no OCR, cloud dependency, multiuser accounts, or automatic online literature ingestion.

**Hardware:** GMKtec mini PC, Ryzen 7 PRO CPU, approximately 32 GB RAM, no NVIDIA GPU. Target Windows/WSL2 or Linux; use CPU-compatible packages. Models may require a one-time download; inference is offline thereafter.

## 2. Non-negotiable principles
- Start with a working retrieval pipeline; add LLM, API, and UI incrementally. **Do not implement all phases at once.**
- Do not use paid APIs. Ollama is the default local LLM provider. Provide an **optional** OpenAI provider behind configuration; it must never be required for tests or startup, and credentials must not be committed.
- Retrieval corpus is restricted to PDFs the user has uploaded. No internet search. Prompt the LLM to use only supplied passages, but **do not claim that a prompt alone guarantees factual grounding**. Provide citations and an insufficient-evidence response.
- Store PDF files on local disk, document/chunk metadata in SQLite, and vectors in FAISS. Maintain stable IDs mapping FAISS results back to chunks, papers, and page numbers. FAISS is a similarity index, **not** a relational database.
- Use plain Python, PyMuPDF, sentence-transformers, FAISS, and Ollama HTTP API initially. Avoid LangChain/LlamaIndex in the first implementation so the data flow is transparent.
- Clear code, type hints, docstrings for nontrivial logic, error handling, logging, pytest, and README. No fake evaluation scores or unsupported claims.
- Treat this as a useful daily research tool, not just a portfolio demo: preserve uploaded PDFs, editable notes, stable source citations, and recoverability.
- Keep `tests/` and `evaluation/` from Phase 1. Require an explicit phase-gate report before progressing. Never self-approve the next phase.

## 3. Architecture

### Ingestion / indexing
`PDF upload -> local PDF file -> PyMuPDF text extraction (per page) -> text cleaning -> chunking with page metadata -> embedding model -> FAISS index`.

Persist in SQLite:
- `papers`: `id`, `title`, `authors` (nullable), `year` (nullable), `doi` (nullable), `file_path`, `file_sha256` (unique), `created_at`.
- `chunks`: `id`, `paper_id` (FK), `page_number`, `chunk_index`, `text`, `embedding_model`, and any metadata needed to map vector IDs.
- `wiki_notes` (later): `id`, `paper_id`, `section`, `content`, `is_ai_generated`, `updated_at`.

Use a durable FAISS ID -> SQLite chunk ID mapping (e.g. `IndexIDMap2` with stable integer chunk IDs). Use atomic or recoverable index updates; on inconsistency provide a `rebuild-index` operation. Avoid duplicate ingestion using SHA-256.

### Query / generation
`Question -> same embedding model -> FAISS top-k search -> resolve chunk text and citations from SQLite -> construct grounded prompt -> local Ollama LLM -> answer + source metadata`.

Ensure retrieval can work **without Ollama**. Implement an explicit no-answer/insufficient-context pathway. The LLM must not fabricate citations: return citations from the retrieval layer and render references as `[Paper title, p. N]`. Treat document text as untrusted content, not instructions.

### Web application (later phases)
`Streamlit frontend -> FastAPI backend -> retrieval/ingestion service -> SQLite + FAISS + Ollama`.

## 4. Recommended stack
- Python 3.11+ (choose a version supported by installed dependencies)
- PyMuPDF for PDF text extraction
- `sentence-transformers/all-MiniLM-L6-v2` for first English embedding baseline (384 dimensions, max 256 word pieces; chunk by tokenizer length, not blindly by characters)
- `faiss-cpu` using normalized embeddings and inner-product index for cosine similarity
- SQLite (`sqlite3`) for metadata
- Ollama for local generation; select a small, available, quantized **1B–3B** model after checking compatibility and memory; make model name configurable, not hard-coded
- FastAPI + Pydantic + Uvicorn (later)
- Streamlit (later)
- pytest; Ruff optional; Docker later (not a Phase 1 prerequisite)

## 5. Stage-gated implementation, testing, and evaluation

### Mandatory workflow for **every** phase
1. **Plan:** State scope, files to modify, test plan, and user-visible behavior before implementation.
2. **Implement only the current phase.** Avoid speculative abstractions and features from future phases.
3. **Automated checks:** Run focused unit/integration tests and existing regression tests. CI tests must not require paid services, internet access, or a running Ollama instance.
4. **Real-user smoke test:** Give copy/paste commands and expected behavior using the user's own permitted PDFs, not just synthetic fixtures. Record any limitations.
5. **Quality evaluation:** Run relevant retrieval/answer metrics where available; record measured values, sample counts, configuration, and errors. Never fabricate outcomes.
6. **Gate report:** Provide a concise `docs/phase-N-report.md` including implemented features, exact commands/tests run, pass/fail, manual checks, known issues, and recommendation to proceed or fix.
7. **Stop and wait for user approval** before starting the next phase. If acceptance fails, repair and rerun checks.

### Phase 1 — PDF ingestion, indexing, retrieval-only CLI (START HERE)
Implement:
1. CLI command to ingest one or more text PDFs from a local folder.
2. Extract page-aware text, skip/report empty pages, clean common whitespace artifacts.
3. Chunk text respecting embedding tokenizer max length; configurable chunk length/overlap; preserve page metadata.
4. Encode chunks on CPU, normalize embeddings, persist FAISS index and SQLite metadata, maintain durable vector-to-chunk mapping.
5. CLI `search "query" --top-k 5` returning ranked excerpts, scores, titles, and page numbers.
6. Duplicate PDF detection via SHA-256; idempotent re-ingestion; fresh-process reload and index recovery/rebuild.
7. Add `tests/` and `evaluation/` immediately; commit a tiny synthetic test corpus, not copyrighted/private papers.

**Automated acceptance:** extraction/page association, tokenizer bounds, deterministic chunking, empty/scanned PDF handling, duplicates, stable IDs, persisted reload, index rebuild, and retrieval ranking on a controlled fixture.

**User smoke test:** ingest at least 2 real, legally held text PDFs; search for a known method/result; open cited PDF pages to verify passages; restart the CLI and repeat; re-ingest and verify unchanged counts.

**Evaluation gate:** create a human-labeled `evaluation/questions.jsonl` with ~20–30 questions, relevant paper/page or chunk references, plus irrelevant/unanswerable queries. Record Recall@1, Recall@5, MRR, page-reference correctness, query latency, and failures. Define ground truth before tuning; hold out some questions for final checks. Do not declare arbitrary target metrics as achieved.

### Phase 2 — Local RAG with Ollama
1. Add configurable Ollama HTTP client and select a CPU-compatible quantized 1B–3B model based on measured performance; support timeouts and service-down errors.
2. Construct prompts from retrieved passages and user question; return answer with retrieval-layer citations and stable paper/page/chunk IDs.
3. Implement cautious no-answer behavior for missing or weak evidence. Calibrate any similarity threshold on evaluation examples; do not assume one universal cutoff.
4. Preserve retrieval-only mode. Treat retrieved PDF text as untrusted data, not executable instructions.

**Automated acceptance:** mocked LLM responses, prompt/citation mapping, timeout/error paths, no-evidence behavior, and regression tests without running Ollama.

**User smoke test:** ask known-answer and out-of-corpus questions about own papers; verify cited pages manually; measure CPU response times.

**Evaluation gate:** manual answer correctness, claim-level faithfulness to cited passages, citation accuracy, abstention on unsupported questions, and latency. Record individual errors and compare to retrieval-only baseline. **Do not assume the model's pretrained knowledge can be completely suppressed.**

### Phase 3 — FastAPI REST backend
Endpoints:
- `GET /health` — health/readiness
- `POST /papers` — multipart PDF upload + ingest
- `GET /papers` and `GET /papers/{paper_id}` — list/details
- `DELETE /papers/{paper_id}` — delete with consistent vector-index update/rebuild
- `GET /search?q=...&top_k=...` — retrieval
- `POST /ask` — question, optional paper filters, answer and citations

Pydantic schemas; appropriate 201/404/422/503 responses; safe upload validation, filename handling, SQL parameters, and localhost binding by default.

**Gate:** API integration tests for success, invalid PDF, duplicates, unknown IDs, deletions, index consistency, malformed inputs, unavailable Ollama. Verify `/docs` and one end-to-end HTTP workflow; rerun retrieval/RAG evaluation to detect regressions.

### Phase 4 — Usable research UI
Streamlit: upload/list/filter papers; paper details and PDF links; ranked semantic search with clickable page citations where feasible; ask questions with source excerpts; select/restrict papers. Handle empty states and errors clearly.

**Gate:** manually complete upload -> search -> ask -> verify page -> restart -> find same paper -> delete -> confirm removed from results. Confirm no accidental data loss, acceptable CPU responsiveness, and readable error messages.

### Phase 5 — Genuine wiki and cross-paper analysis
Implement editable **user-authored** paper notes (research question, methods, findings, limitations), optional clearly labeled AI-generated drafts, topic/concept pages, bidirectional links or backlinks between pages, and grounded comparisons across selected papers. A collection of PDF summaries alone is a library, not yet a wiki.

**Gate:** verify note create/edit/reopen, link navigation, user/AI content provenance, and source-grounded comparison with correct page citations. No automatic overwriting of user edits.

### Phase 6 — Deployment, CI, and portfolio readiness
- Add GitHub Actions CI with offline unit tests and lightweight fixtures; use mocks at model boundaries.
- Dockerize after the native app works; preserve SQLite/index/PDF data in mounted volumes and document backups.
- Add structured logs, basic timings, reproducible setup, architecture diagram, screenshots, and real evaluation report.
- Never commit personal PDFs, API keys, model weights, or runtime databases.

**Gate:** clean install from README, successful test suite, persistent data across container restarts, documented limitations, and reproducible evaluation commands.

### Optional Phase 7 — Agentic research assistant (AFTER reliable RAG)
Extend, rather than replace, the tested retrieval system. Add a **single tool-using agent** that can choose between searching selected papers, reading specific passages, querying metadata, and drafting comparisons/notes. Require confirmation before any persistent edit or deletion. Add bounded tool calls, trace logs, timeouts, source verification, and tests for tool choice, loop termination, prompt injection, and failed tools. Consider LangGraph/MCP only if a concrete need arises; avoid multi-agent complexity by default.

**Gate:** compare agent answers against the non-agent RAG baseline on a fixed multi-step benchmark; report tool-call count, correctness, citation quality, latency, and failures. Do not call ordinary fixed RAG an agent.

## 6. Suggested repository structure
```text
papertrail/
  README.md
  PROJECT_SPEC.md
  .env.example
  .gitignore
  pyproject.toml
  data/                 # gitignored: uploaded PDFs, DB, index
  src/papertrail/
    config.py
    ingestion.py        # PDF extraction and dedup
    chunking.py
    embeddings.py
    vector_store.py     # FAISS persistence and ID mapping
    database.py         # SQLite schema and queries
    retrieval.py
    rag.py              # Phase 2
    llm_providers.py    # Phase 2
    api.py              # Phase 3
    ui.py               # Phase 4
    cli.py
  tests/
    test_chunking.py
    test_database.py
    test_ingestion.py
    test_retrieval.py
    test_api.py
    fixtures/
  evaluation/
    questions.example.jsonl
    questions.jsonl        # local annotations; no private paper text in Git
    evaluate.py
    reports/
  docs/
    phase-1-report.md      # generated only after checks actually run
    phase-2-report.md      # future
```

## 7. Configuration
Example `.env.example`:
```dotenv
DATA_DIR=./data
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
CHUNK_TOKENS=180
CHUNK_OVERLAP_TOKENS=30
TOP_K=5
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=CHANGE_TO_INSTALLED_MODEL
OPENAI_API_KEY=
OPENAI_MODEL=
```
Token values are starting points, not proven optimal. Do not include `.env`, PDFs, SQLite DB, FAISS files, model weights, or user data in Git.

## 8. Engineering notes
- Extract and chunk by page to preserve page-level citation accuracy. Long sections may span pages, but preserve start/end page IDs.
- Sentence-transformers model `all-MiniLM-L6-v2` truncates inputs above its tokenizer limit; verify chunk token lengths before encoding.
- For FAISS cosine search, L2-normalize both document and query embeddings and use inner-product similarity.
- Use SQLite transactions for metadata changes and a recoverable strategy for FAISS/index persistence (they are not one atomic transaction).
- SQLite is adequate for single-user local MVP; don't introduce PostgreSQL unless needed.
- Don't imply that a local app must have a Web API: the retrieval-only CLI is intentionally API-free.
- Optional OpenAI provider is a future extension, never a dependency of the core implementation.

## 9. Instructions to Codex — first task
**Implement Phase 1 only and stop for approval.** Read the entire specification. First propose a minimal package layout, CLI commands, acceptance tests, and evaluation protocol. Then implement page-aware PDF ingestion, CPU token-aware embeddings, FAISS, SQLite, persistence, deduplication, and semantic-search CLI. Create `tests/` and `evaluation/` now. Use synthetic fixtures for automated tests; provide instructions for a smoke test using the user's own PDFs. Run actual tests, show commands and outcomes, and write `docs/phase-1-report.md` with factual evidence (never invent scores). If real PDFs or human ground-truth labels are unavailable, mark those checks **PENDING USER VALIDATION**, not PASS. Do not implement Ollama, FastAPI, Streamlit, agent tooling, or Docker in Phase 1. Do not start Phase 2 without explicit user approval. Write educational comments explaining component boundaries and data flow.

## 9.1 Suggested user workflow after each Codex phase
1. Review Codex's phase report and run its listed smoke-test commands yourself.
2. Verify 2–3 actual source PDF pages for retrieval/citation correctness.
3. Share failures or confusing behavior with Codex and request fixes plus rerun.
4. Only after evidence is satisfactory, explicitly say: `Proceed to Phase N`.

## 10. Reference documentation
- Sentence Transformers model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Ollama API: https://docs.ollama.com/api
- FAISS: https://github.com/facebookresearch/faiss
- FastAPI: https://fastapi.tiangolo.com/
- PyMuPDF: https://pymupdf.readthedocs.io/
