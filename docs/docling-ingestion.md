# Docling ingestion experiment

## What ingestion means

Ingestion is the preparation step that turns local PDFs into a searchable
library. PaperTrail validates and copies each PDF, extracts text and structure,
splits the document into chunks, embeds those chunks, and saves metadata in
SQLite plus vectors in FAISS. It does not generate an answer at this stage.

## Why two backends remain

`pymupdf` is the default lightweight backend. PyMuPDF is the Python binding for
MuPDF. It reads and renders PDFs and exposes text, coordinates, and blocks very
quickly. A PDF, however, commonly stores positioned glyphs rather than semantic
objects such as “Methods section” or “this cell belongs to this table header.”
PyMuPDF does not run a learned layout model, so multi-column order, headings,
captions, and tables can be ambiguous.

`docling` is an optional structured backend. It applies document layout and
table analysis, then uses Docling's `HybridChunker`. This is the Python
equivalent of the structure-aware workflow demonstrated by `doclingr`'s
`docling_chunk()`; it is not a call from Python into the R package. The cost is
a larger installation, an initial model download, and slower CPU ingestion.

## MiniLM decision

`sentence-transformers/all-MiniLM-L6-v2` came from the original PaperTrail
project specification. It is a practical Phase 1 baseline because it is a
small English sentence encoder, runs acceptably on CPU, and produces compact
384-dimensional vectors. It was not selected because it is universally best.
The evaluation results are the test of that choice, and a later experiment can
compare domain-specific or retrieval-trained encoders without changing the
storage interface.

## Install and run

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[model,docling,test]'
$env:HF_HOME = Join-Path (Get-Location) 'data/model-cache'
.\.venv\Scripts\papertrail.exe --data-dir data/docling180 ingest . `
  --backend docling --chunk-tokens 180 --docling-threads 4
.\.venv\Scripts\papertrail.exe --data-dir data/docling180 search `
  "How many patients developed anal cancer during follow-up?" --top-k 5
```

Use a separate data directory when comparing backends. Duplicate detection is
based on PDF SHA-256, so an already ingested file is intentionally not parsed a
second time inside the same library.

## Parameters

| Parameter | Current value | Meaning |
|---|---:|---|
| `--backend` | `pymupdf` by default | Selects fast page extraction or Docling structure analysis. |
| `--chunk-tokens` | `180` | Maximum token budget given to the chunker. For Docling, headings added for embedding must also fit this budget. |
| `--overlap` | `30` | Repeats boundary tokens for PyMuPDF chunks only. Docling follows document structure instead. |
| `--docling-merge-peers` | enabled | Merges adjacent small chunks when they have compatible metadata/headings. |
| `--docling-repeat-table-header` | enabled | Repeats table headers when a large table is split so cells retain column meaning. |
| `--docling-threads` | `4` | CPU worker count for parsing; conservative for a laptop. |
| OCR | disabled in code | The Phase 1 PDFs contain embedded text. Scanned PDFs are outside this experiment. |
| Table structure | enabled in code | Preserves scientific table relationships instead of treating them as arbitrary lines. |
| Accelerator | CPU in code | Keeps the portfolio reproducible without a GPU. |

The Docling tokenizer is loaded from the same MiniLM model used to create the
embeddings. PaperTrail saves raw chunk text for inspection/citation and a
separate heading-contextualized text for embedding.

## Two-PDF validation

The first Docling run processed both PDFs in 110.37 seconds on CPU and produced
457 chunks (326 for the systematic review and 131 for the cohort paper). Of
those, 456 retained non-empty heading metadata. Top-5 retrieval was manually
judged by checking the returned text for the expected evidence, not by accepting
a matching page number.

| Pipeline | Hit@1 | Hit@5 | MRR |
|---|---:|---:|---:|
| PyMuPDF block-order, 180/30 | 4/18 (0.2222) | 8/18 (0.4444) | 0.3056 |
| Docling HybridChunker, 180 | 3/18 (0.1667) | 9/18 (0.5000) | 0.2741 |

Docling improved top-5 coverage by one question but reduced early-rank quality.
The main failures were references outranking article content, scientific terms
shared by both papers causing cross-document matches, exact numeric facts not
matching the dense query strongly, and cross-paper questions needing evidence
from more than one document. Structure-aware chunking alone therefore does not
solve retrieval ranking.

Recommended next experiments are section/reference filtering, hybrid BM25 plus
dense retrieval, document-aware routing or metadata filters, and a reranker.
These should be evaluated one at a time against the same gold set.
