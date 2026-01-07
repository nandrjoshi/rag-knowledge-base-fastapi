from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    content: str


def chunk_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """
    Split text into overlapping character-based chunks.

    Rules:
    - chunk_size must be > 0
    - chunk_overlap must be >= 0 and < chunk_size
    - Chunks are trimmed; empty chunks are dropped.
    - Deterministic ordering by chunk_index.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    text = (text or "").strip()
    if not text:
        return []

    chunks: list[TextChunk] = []
    step = chunk_size - chunk_overlap

    start = 0
    idx = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(TextChunk(chunk_index=idx, content=chunk))
            idx += 1
        if end == n:
            break
        start += step

    return chunks
