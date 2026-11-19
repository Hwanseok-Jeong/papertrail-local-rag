"""SQLite owns document records. FAISS is a rebuildable search index."""

import hashlib
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import faiss
import numpy as np
import pymupdf

from .embeddings import Encoder, split_page


class Library:
    def __init__(self, directory: Path, encoder: Encoder, length: int = 180, overlap: int = 30):
        self.directory = directory
        self.encoder = encoder
        self.length = length
        self.overlap = overlap
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "pdfs").mkdir(exist_ok=True)
        self.index_path = directory / "chunks.faiss"

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.directory / "papertrail.sqlite", timeout=60)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY, title TEXT NOT NULL, authors TEXT,
                    year INTEGER, doi TEXT, file_path TEXT NOT NULL,
                    file_sha256 TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL REFERENCES papers(id),
                    page_number INTEGER NOT NULL, chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL, embedding_model TEXT NOT NULL,
                    UNIQUE (paper_id, page_number, chunk_index)
                );
            """)
            # All CLI readers and writers share this lock while using the index.
            db.execute("BEGIN IMMEDIATE")
            yield db

    def vectors(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(self.encoder.encode(texts), dtype=np.float32, order="C", copy=True)
        if vectors.shape != (len(texts), self.encoder.dimension):
            raise ValueError("Encoder returned unexpected vector dimensions")
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Encoder returned invalid or zero-length vectors")
        faiss.normalize_L2(vectors)
        return vectors

    def load_index(self, db: sqlite3.Connection, rebuild: bool = False) -> Any:
        rows = db.execute("SELECT id, embedding_model FROM chunks ORDER BY id").fetchall()
        if any(row["embedding_model"] != self.encoder.name for row in rows):
            raise ValueError("Use the original embedding model or a separate data directory")
        ids = np.array([row["id"] for row in rows], dtype=np.int64)
        if not rebuild and self.index_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                if (
                    isinstance(index, faiss.IndexIDMap2)
                    and index.d == self.encoder.dimension
                    and index.metric_type == faiss.METRIC_INNER_PRODUCT
                    and np.array_equal(np.sort(faiss.vector_to_array(index.id_map)), ids)
                ):
                    return index
            except RuntimeError:
                # A missing or unreadable index can be recreated from stored passages.
                pass
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.encoder.dimension))
        if len(ids):
            cursor = db.execute("SELECT id, text FROM chunks ORDER BY id")
            while batch := cursor.fetchmany(128):
                index.add_with_ids(
                    self.vectors([row["text"] for row in batch]),
                    np.array([row["id"] for row in batch], dtype=np.int64),
                )
        self.save_index(index)
        return index

    def save_index(self, index: Any) -> None:
        # Replace the index only after FAISS finishes writing the new file.
        with NamedTemporaryFile(dir=self.directory, delete=False) as temporary:
            path = Path(temporary.name)
        try:
            faiss.write_index(index, str(path))
            os.replace(path, self.index_path)
        finally:
            path.unlink(missing_ok=True)

    def ingest(self, path: Path) -> dict[str, Any]:
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM papers WHERE file_sha256 = ?", (sha,)).fetchone():
                return {"file": path.name, "status": "duplicate"}
            chunks = []
            empty_pages = []
            with pymupdf.open(stream=content, filetype="pdf") as pdf:
                if not pdf.is_pdf or pdf.needs_pass:
                    raise ValueError("Expected a readable, unencrypted PDF")
                metadata = pdf.metadata or {}
                title = (metadata.get("title") or "").strip() or path.stem
                authors = (metadata.get("author") or "").strip() or None
                for page_number, page in enumerate(pdf, start=1):
                    page_text = re.sub(r"\s+", " ", page.get_text(sort=True)).strip()
                    passages = split_page(page_text, self.encoder, self.length, self.overlap)
                    if not passages:
                        empty_pages.append(page_number)
                    chunks.extend((page_number, i, text) for i, text in enumerate(passages))
            if not chunks:
                return {"file": path.name, "status": "no_text", "empty_pages": empty_pages}

            index = self.load_index(db)
            relative_path = Path("pdfs") / f"{sha}.pdf"
            with NamedTemporaryFile(dir=self.directory / "pdfs", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(content)
            try:
                os.replace(temporary_path, self.directory / relative_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            paper_id = db.execute(
                "INSERT INTO papers(title, authors, file_path, file_sha256) VALUES (?, ?, ?, ?)",
                (title, authors, relative_path.as_posix(), sha),
            ).lastrowid
            for start in range(0, len(chunks), 128):
                batch = chunks[start:start + 128]
                ids = [
                    db.execute(
                        """INSERT INTO chunks(paper_id, page_number, chunk_index, text, embedding_model)
                           VALUES (?, ?, ?, ?, ?)""",
                        (paper_id, *chunk, self.encoder.name),
                    ).lastrowid
                    for chunk in batch
                ]
                index.add_with_ids(
                    self.vectors([chunk[2] for chunk in batch]), np.array(ids, dtype=np.int64)
                )
            # If SQLite rolls back after replacement, the ID check repairs FAISS on next use.
            self.save_index(index)
            return {
                "file": path.name, "status": "ingested", "paper_id": paper_id,
                "chunks": len(chunks), "empty_pages": empty_pages,
            }

    def rebuild(self) -> int:
        with self.transaction() as db:
            return self.load_index(db, rebuild=True).ntotal

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1:
            raise ValueError("Search requires a nonblank query and positive top-k")
        with self.transaction() as db:
            index = self.load_index(db)
            if not index.ntotal:
                return []
            scores, ids = index.search(self.vectors([query.strip()]), min(top_k, index.ntotal))
            results = []
            for score, chunk_id in zip(scores[0], ids[0], strict=True):
                row = db.execute(
                    """SELECT chunks.id AS chunk_id, paper_id, page_number, text,
                              title, file_path, file_sha256
                       FROM chunks JOIN papers ON papers.id = chunks.paper_id
                       WHERE chunks.id = ?""",
                    (int(chunk_id),),
                ).fetchone()
                results.append(dict(row, score=float(score)))
            return results
