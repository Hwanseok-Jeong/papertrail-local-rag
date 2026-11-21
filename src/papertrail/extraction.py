"""Extract citation-aware chunks with either the lightweight or structured backend."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .embeddings import Encoder, split_page


@dataclass(frozen=True)
class ExtractedChunk:
    """Keep display text separate from the heading-enriched embedding text."""

    page_start: int
    page_end: int
    raw_text: str
    embedding_text: str
    headings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractedDocument:
    title: str
    authors: str | None
    chunks: tuple[ExtractedChunk, ...]
    empty_pages: tuple[int, ...]


def extract_pymupdf(
    path: Path, content: bytes, encoder: Encoder, length: int, overlap: int
) -> ExtractedDocument:
    """Extract page text quickly and split it with fixed token overlap."""
    chunks: list[ExtractedChunk] = []
    empty_pages: list[int] = []
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        if not pdf.is_pdf or pdf.needs_pass:
            raise ValueError("Expected a readable, unencrypted PDF")
        metadata = pdf.metadata or {}
        title = (metadata.get("title") or "").strip() or path.stem
        authors = (metadata.get("author") or "").strip() or None
        for page_number, page in enumerate(pdf, start=1):
            blocks = page.get_text("blocks", sort=False)
            page_text = " ".join(" ".join(b[4] for b in blocks if b[6] == 0).split())
            passages = split_page(page_text, encoder, length, overlap)
            if not passages:
                empty_pages.append(page_number)
            chunks.extend(
                ExtractedChunk(page_number, page_number, text, text)
                for text in passages
            )
    return ExtractedDocument(title, authors, tuple(chunks), tuple(empty_pages))


def extract_docling(
    path: Path,
    content: bytes,
    encoder: Encoder,
    length: int,
    *,
    merge_peers: bool,
    repeat_table_header: bool,
    threads: int,
) -> ExtractedDocument:
    """Recover document structure and create tokenizer-aware hybrid chunks."""
    try:
        from docling.chunking import HybridChunker
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    except ImportError as error:
        raise ImportError("Install Docling with: pip install -e '.[docling]'") from error

    # PyMuPDF remains the inexpensive source of file metadata and empty-page checks.
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        if not pdf.is_pdf or pdf.needs_pass:
            raise ValueError("Expected a readable, unencrypted PDF")
        metadata = pdf.metadata or {}
        title = (metadata.get("title") or "").strip() or path.stem
        authors = (metadata.get("author") or "").strip() or None
        empty_pages = tuple(
            page_number
            for page_number, page in enumerate(pdf, start=1)
            if not page.get_text().strip()
        )

    pipeline = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU,
            num_threads=threads,
        ),
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)},
        show_progress=False,
    )
    result = converter.convert(path)
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name=encoder.name,
        max_tokens=length,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=merge_peers,
        repeat_table_header=repeat_table_header,
    )

    chunks: list[ExtractedChunk] = []
    for chunk in chunker.chunk(result.document):
        raw_text = chunk.text.strip()
        embedding_text = chunker.contextualize(chunk).strip()
        pages = sorted({
            provenance.page_no
            for item in chunk.meta.doc_items
            for provenance in (getattr(item, "prov", None) or [])
            if provenance.page_no is not None
        })
        if not raw_text or not pages:
            continue
        if encoder.token_count(embedding_text) > encoder.max_tokens:
            raise ValueError("Docling produced a chunk beyond the embedding model token limit")
        chunks.append(ExtractedChunk(
            page_start=pages[0],
            page_end=pages[-1],
            raw_text=raw_text,
            embedding_text=embedding_text,
            headings=tuple(chunk.meta.headings or ()),
        ))
    return ExtractedDocument(title, authors, tuple(chunks), empty_pages)
