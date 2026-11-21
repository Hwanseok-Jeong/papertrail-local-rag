# Phase 2 — Local grounded answers

## Delivered

- The ask CLI retrieves five passages with the Phase 1 library and sends them to a configurable local Ollama model.
- The model must return structured claims with a SQLite chunk ID and an exact evidence quote.
- Application code resolves each accepted chunk to its paper title and PDF page.
- Empty claims, malformed output, unknown chunks, invented quotes, and introduced name-like values absent from the quote return insufficient evidence.
- Search remains available without Ollama.
- The evaluation runner saves each question, expected answer, retrieved passages, generated answer, citations, and elapsed time as JSONL.

## Code map

| File | Responsibility |
| --- | --- |
| src/papertrail/rag.py | Ollama HTTP request, grounded prompt, structured response schema, claim validation, and citation rendering |
| src/papertrail/library.py | PDF extraction, SQLite and FAISS persistence, and retrieval used by both search and ask |
| src/papertrail/cli.py | ask and search command-line entry points |
| evaluation/evaluate_answers.py | Reproducible batch run over the supplied gold CSV |
| tests/test_rag.py | Mocked generation, attribution, rejection, timeout, and service failure tests |

## Data flow

1. Library.search embeds the question and returns the top five SQLite-backed chunks from FAISS.
2. rag.generate sends only the question and those excerpts to Ollama. The prompt treats excerpts as untrusted data.
3. Ollama returns JSON claims. Each claim supplies answer text, a chunk ID, and an exact quote.
4. rag.ask checks the ID, quote, statement, and introduced values. It constructs citations from retrieval metadata rather than model-written page labels.
5. A validation failure rejects the complete response with insufficient evidence.

## Automated validation

Command:

    .\.venv\Scripts\python.exe -m pytest -q

Result: 11 passed. Tests do not need Ollama or network access. They cover prompt/schema construction, source and page mapping, invented chunk and quote rejection, invented named values, timeouts, and unavailable service.

## Real CPU evaluation

Environment:

- Windows CPU execution
- Ollama 0.40.16
- Qwen 2.5 1.5B and 3B, Q4_K_M
- all-MiniLM-L6-v2 retrieval, top five
- Two supplied papers and 22 supplied questions

The strict manual criterion requires a complete answer to the question. Exact quote matching alone does not establish correctness.

| Model | Fully correct answerable | Correct negative abstention | Mean time | Range |
| --- | ---: | ---: | ---: | ---: |
| Qwen 2.5 1.5B | 2/18 | 4/4 | 15.7 s | 6.4–26.0 s |
| Qwen 2.5 3B | 3/18 | 3/4 | 27.2 s | 12.6–51.2 s |

The 3B correct answers were A01, A04, and B05. It gave partial answers for A08 and B04. It answered A02, A03, B02, B03, and B07 incorrectly. The full comparison questions were unsupported by the retrieved context and correctly produced no answer.

Before the introduced-value check, N01 invented MinION while citing an unrelated passage. The new check rejected N01 in a targeted rerun. A targeted A01 rerun then abstained, showing that this small model is nondeterministic in practice and the added check increases false abstention. The complete 22-question run has not been repeated after this final guard.

Raw evaluation records are local and gitignored:

- data/phase2-qwen15b.jsonl
- data/phase2-qwen3b.jsonl

## Gate assessment

The Phase 2 implementation is operational and tested, but answer quality does not pass a daily-use quality gate on this development set. Retrieval remains the dominant limit for several questions, while the small models also misread numbers, return incomplete multi-part answers, and sometimes attach a real quote to an unsupported claim.

Recommended work before Phase 3:

- Improve Phase 1 retrieval with a held-out evaluation set, especially lexical/dense fusion and reference-section suppression.
- Add an explicit entailment verifier or use a stronger local model after measuring CPU cost.
- Repeat the full answer evaluation after validation changes.
- Calibrate abstention with development data, then confirm it on held-out questions.

Phase 3 requires explicit approval.
