"""Synthetic PDF fixtures exercise persistence without downloading a model."""

import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from papertrail.embeddings import split_page
from papertrail.library import Library
from papertrail.extraction import ExtractedChunk, ExtractedDocument


class FixtureEncoder:
    name = "fixture"
    dimension = 2
    max_tokens = 8

    def offsets(self, text):
        return [match.span() for match in re.finditer(r"\S+", text)]

    def token_count(self, text):
        return len(self.offsets(text)) + 2

    def encode(self, texts):
        return np.array([[2, 0] if "orchid" in text else [0, 3] for text in texts])


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "synthetic.pdf"
    with pymupdf.open() as pdf:
        pdf.set_metadata({"title": "Synthetic Study"})
        pdf.new_page().insert_text((50, 50), "Granite forms in the crust.")
        pdf.new_page()
        pdf.new_page().insert_text((50, 50), "An orchid grew in shade.")
        pdf.save(path)
    return path


def test_chunking():
    encoder = FixtureEncoder()
    text = "a b c d e f g"
    assert split_page(text, encoder, 4, 1) == ["a b c d", "d e f g"]
    assert all(encoder.token_count(chunk) <= encoder.max_tokens
               for chunk in split_page(text, encoder, 6, 1))
    with pytest.raises(ValueError):
        split_page(text, encoder, 7, 1)


def test_ingest_reload_recovery_and_rank(tmp_path, sample_pdf):
    directory = tmp_path / "data"
    encoder = FixtureEncoder()
    library = Library(directory, encoder, 6, 1)
    result = library.ingest(sample_pdf)
    assert result["empty_pages"] == [2]
    assert result["chunks"] == 2
    duplicate = sample_pdf.with_name("renamed.pdf")
    duplicate.write_bytes(sample_pdf.read_bytes())
    assert library.ingest(duplicate)["status"] == "duplicate"

    results = Library(directory, encoder).search("orchid", 5)
    assert [row["page_number"] for row in results] == [3, 1]
    assert results[0]["title"] == "Synthetic Study"
    assert results[0]["score"] == pytest.approx(1)
    assert (directory / results[0]["file_path"]).read_bytes() == sample_pdf.read_bytes()

    # A fresh Python interpreter verifies that retrieval survives process exit.
    test_module = Path(__file__).resolve()
    script = (
        "import sys; from pathlib import Path; "
        "sys.path[:0] = ['src', str(Path(sys.argv[1]).parent)]; "
        "from test_library import FixtureEncoder, Library; "
        "print(Library(Path(sys.argv[2]), FixtureEncoder()).search('orchid')[0]['chunk_id'])"
    )
    run = subprocess.run(
        [sys.executable, "-c", script, str(test_module), str(directory)],
        check=True, capture_output=True, text=True,
    )
    assert int(run.stdout) == results[0]["chunk_id"]

    library.index_path.write_bytes(b"invalid index")
    assert library.search("orchid") == results
    library.index_path.unlink()
    assert library.rebuild() == 2
    assert library.search("orchid") == results


def test_blank_pdf(tmp_path):
    path = tmp_path / "blank.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(path)
    directory = tmp_path / "data"
    library = Library(directory, FixtureEncoder(), 6, 1)
    assert library.ingest(path)["status"] == "no_text"
    assert library.search("orchid") == []
    with sqlite3.connect(directory / "papertrail.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0


def test_recovers_index_ahead_of_sqlite(tmp_path, sample_pdf, monkeypatch):
    library = Library(tmp_path / "data", FixtureEncoder(), 6, 1)
    save_index = library.save_index

    def interrupt_after_replace(index):
        save_index(index)
        if index.ntotal:
            raise OSError("interrupted before SQLite commit")

    with monkeypatch.context() as patch:
        patch.setattr(library, "save_index", interrupt_after_replace)
        with pytest.raises(OSError, match="interrupted"):
            library.ingest(sample_pdf)

    assert library.search("orchid") == []
    assert library.ingest(sample_pdf)["chunks"] == 2
    assert library.search("orchid")[0]["page_number"] == 3


def test_docling_backend_stores_structure_and_embeds_context(
    tmp_path, sample_pdf, monkeypatch
):
    document = ExtractedDocument(
        title="Structured Study",
        authors="A. Researcher",
        chunks=(ExtractedChunk(
            page_start=2,
            page_end=3,
            raw_text="The visible evidence passage.",
            embedding_text="orchid Methods The visible evidence passage.",
            headings=("Methods",),
        ),),
        empty_pages=(),
    )
    captured = {}

    def fake_extract(path, content, encoder, length, **settings):
        captured.update(settings)
        return document

    monkeypatch.setattr("papertrail.library.extract_docling", fake_extract)
    library = Library(
        tmp_path / "docling-data",
        FixtureEncoder(),
        length=6,
        backend="docling",
        docling_merge_peers=False,
        docling_repeat_table_header=True,
        docling_threads=2,
    )
    result = library.ingest(sample_pdf)
    hit = library.search("orchid", 1)[0]

    assert result["backend"] == "docling"
    assert captured == {
        "merge_peers": False,
        "repeat_table_header": True,
        "threads": 2,
    }
    assert hit["text"] == "The visible evidence passage."
    assert hit["page_number"] == 2
    assert hit["page_number_end"] == 3
    assert hit["headings"] == ["Methods"]
    assert hit["ingestion_backend"] == "docling"
