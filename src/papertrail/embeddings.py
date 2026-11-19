"""Token-aware text splitting and CPU embedding."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Encoder(Protocol):
    name: str
    dimension: int
    max_tokens: int

    def offsets(self, text: str) -> Sequence[tuple[int, int]]: ...
    def token_count(self, text: str) -> int: ...
    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...


class MiniLM:
    def __init__(self, name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.model = SentenceTransformer(name, device="cpu")
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()
        self.max_tokens = self.model.max_seq_length

    def offsets(self, text: str) -> Sequence[tuple[int, int]]:
        return self.model.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )["offset_mapping"]

    def token_count(self, text: str) -> int:
        return len(self.model.tokenizer(text, verbose=False)["input_ids"])

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def split_page(text: str, encoder: Encoder, length: int, overlap: int) -> list[str]:
    """Keep original page excerpts and check bounds after retokenizing each slice."""
    if not 0 <= overlap < length <= encoder.max_tokens - encoder.token_count(""):
        raise ValueError("Require 0 <= overlap < chunk size within the encoder token budget")
    offsets = encoder.offsets(text)
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + length, len(offsets))
        # A slice starting inside a word can tokenize differently from the full page.
        while end > start:
            excerpt = text[offsets[start][0]:offsets[end - 1][1]]
            if encoder.token_count(excerpt) <= encoder.max_tokens:
                break
            end -= 1
        if end == start:
            raise ValueError("A page token could not fit within the encoder token limit")
        chunks.append(excerpt)
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)
    return chunks
