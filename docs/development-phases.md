# PaperTrail development phases and validation gates

PaperTrail is developed in reviewable phases. Each phase adds one operational layer and retains the tests and evaluation evidence from earlier phases.

## Phase 1 — PDF retrieval foundation

**Purpose.** Build a transparent local search system before adding generated answers.

**Implementation.**

- Import text-based PDFs, preserve page numbers, report empty pages, and prevent duplicates with SHA-256.
- Split page text within the embedding model's token limit.
- Store papers and chunks in SQLite. Store normalized vectors in FAISS with stable chunk IDs.
- Provide ingest, search, and rebuild-index commands.

**Validation.**

- Automated tests cover extraction, page association, token bounds, duplicate import, stable IDs, restart persistence, ranking, and index recovery.
- Real-PDF evaluation verifies the actual evidence inside each retrieved chunk. It records Hit@1, Hit@5, MRR, document/page/chunk IDs, scores, excerpts, and latency.
- Reimport must leave paper and chunk counts unchanged. A fresh process must reproduce search results.

## Phase 2 — Local grounded answers

**Purpose.** Turn retrieved passages into readable answers while keeping every accepted claim traceable to the local corpus.

**Implementation.**

- Call a configurable local Ollama model with the question and top retrieved chunks.
- Require structured claims containing a source chunk ID and an exact evidence quote.
- Resolve citations in application code as [paper title, p. N].
- Return insufficient_evidence for empty claims, unknown source IDs, invalid quotes, malformed output, or incomplete evidence.
- Keep retrieval-only commands available.

**Validation.**

- Offline tests mock Ollama and cover prompt construction, citation mapping, malformed or invented evidence, timeouts, and unavailable service.
- Run all answerable gold questions plus the four unsupported questions on the local CPU model.
- Manually score answer correctness, claim support, citation accuracy, abstention, and latency. Compare failures with the Phase 1 retrieval results to identify retrieval and generation errors separately.

## Phase 3 — FastAPI backend

**Purpose.** Expose the tested library and RAG operations through a stable local API.

**Implementation.**

- Add health, upload, paper list/detail/delete, search, and ask endpoints.
- Validate uploads and filenames, use parameterized SQL, and return explicit HTTP errors.
- Keep deletion and FAISS state consistent, rebuilding when required.

**Validation.**

- API integration tests cover valid and invalid PDFs, duplicates, missing papers, deletion, malformed input, index consistency, and unavailable Ollama.
- Exercise one complete HTTP workflow and inspect OpenAPI documentation.
- Rerun retrieval and answer evaluation through the API to detect regressions.

## Phase 4 — Research user interface

**Purpose.** Make paper upload, discovery, grounded questions, and source verification usable in daily research.

**Implementation.**

- Add a Streamlit interface for upload, paper browsing, semantic search, questions, paper filters, excerpts, and page citations.
- Show empty states, service errors, and local source links clearly.

**Validation.**

- Complete upload → search → ask → open cited page → restart → reopen paper → delete.
- Verify persistence, deletion, readable errors, citation navigation, and acceptable CPU response time.

## Phase 5 — Literature wiki and cross-paper analysis

**Purpose.** Add durable research knowledge around the document collection.

**Implementation.**

- Store editable user notes, clearly labeled generated drafts, topic pages, links, backlinks, and comparisons across selected papers.
- Preserve user edits and record whether content was written by the user or generated.

**Validation.**

- Create, edit, close, and reopen notes.
- Follow links in both directions and verify provenance.
- Check every comparison claim against the cited pages and ensure generated content never overwrites user content.

## Phase 6 — Reproducible delivery

**Purpose.** Make the project installable, testable, recoverable, and presentable as a portfolio project.

**Implementation.**

- Add offline CI, structured logging and timings, Docker packaging, persistent volumes, backup guidance, architecture documentation, screenshots, and measured evaluation reports.

**Validation.**

- Install from the README in a clean environment and run all offline tests.
- Restart native and container deployments while preserving PDFs, SQLite, FAISS, and notes.
- Reproduce evaluation commands and confirm that private PDFs, credentials, model weights, and runtime databases are absent from Git.

## Optional Phase 7 — Tool-using research assistant

**Purpose.** Support bounded multi-step research tasks after the fixed RAG baseline is reliable.

**Implementation.**

- Add one agent that can search selected papers, inspect passages, query metadata, and draft comparisons or notes.
- Require confirmation for persistent edits and deletion. Limit tool calls and record traces.

**Validation.**

- Compare the agent with the non-agent RAG baseline on a fixed multi-step benchmark.
- Measure correctness, citation quality, tool-call count, latency, loop termination, prompt-injection handling, and failed-tool recovery.
