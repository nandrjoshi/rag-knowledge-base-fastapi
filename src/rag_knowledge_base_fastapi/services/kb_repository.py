from __future__ import annotations

import json

from sqlalchemy import bindparam, text

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from rag_knowledge_base_fastapi.services.db import create_db_engine
from rag_knowledge_base_fastapi.services.openai_embeddings import OpenAIEmbeddingsClient

EMBED_DIM = 1536  # matches text-embedding-3-small


@dataclass(frozen=True)
class InsertResult:
    inserted: int


def _zero_vector_literal(dim: int) -> str:
    """
    Returns a pgvector literal representing a zero vector of the given dimension.
    Example: '[0,0,0]' for dim=3
    """
    return "[" + ",".join(["0"] * dim) + "]"


def insert_chunks_without_embeddings(
    *,
    source: str,
    doc_id: str | None,
    chunks: list[tuple[int, str]],
    metadata: dict[str, Any] | None = None,
) -> InsertResult:
    """
    Insert chunks into kb_chunks using a placeholder zero-vector embedding.
    This is a temporary step to validate ingestion end-to-end before OpenAI is wired in.

    chunks: list of (chunk_index, content)
    """
    if not source.strip():
        raise ValueError("source is required")
    if not chunks:
        return InsertResult(inserted=0)

    meta = metadata or {}
    zero_vec = _zero_vector_literal(EMBED_DIM)

    engine = create_db_engine()

    sql = (
        text(
            """
            INSERT INTO kb_chunks (source, doc_id, chunk_index, content, metadata, embedding)
            VALUES (:source, :doc_id, :chunk_index, :content,
                    CAST(:metadata AS jsonb),
                    CAST(:embedding AS vector))
            """
        )
        .bindparams(
            bindparam("metadata"),
            bindparam("embedding"),
        )
    )


    with engine.begin() as conn:
        for chunk_index, content in chunks:
            conn.execute(
                sql,
                {
                    "source": source,
                    "doc_id": doc_id,
                    "chunk_index": int(chunk_index),
                    "content": content,
                    "metadata": json.dumps(meta),
                    "embedding": zero_vec,
                },
            )

    return InsertResult(inserted=len(chunks))

def insert_chunks_with_embeddings(
    *,
    source: str,
    doc_id: str | None,
    chunks: list[tuple[int, str]],
    metadata: dict[str, Any] | None = None,
) -> InsertResult:
    """
    Insert chunks into kb_chunks using real OpenAI embeddings.

    chunks: list of (chunk_index, content)
    """
    if not source.strip():
        raise ValueError("source is required")
    if not chunks:
        return InsertResult(inserted=0)

    meta = metadata or {}
    engine = create_db_engine()

    embedder = OpenAIEmbeddingsClient()

    sql = (
        text(
            """
            INSERT INTO kb_chunks (source, doc_id, chunk_index, content, metadata, embedding)
            VALUES (:source, :doc_id, :chunk_index, :content,
                    CAST(:metadata AS jsonb),
                    CAST(:embedding AS vector))
            """
        )
        .bindparams(
            bindparam("metadata"),
            bindparam("embedding"),
        )
    )

    with engine.begin() as conn:
        for chunk_index, content in chunks:
            emb = embedder.embed_text(content)
            conn.execute(
                sql,
                {
                    "source": source,
                    "doc_id": doc_id,
                    "chunk_index": int(chunk_index),
                    "content": content,
                    "metadata": json.dumps(meta),
                    "embedding": "[" + ",".join(map(str, emb.vector)) + "]",
                },
            )

    return InsertResult(inserted=len(chunks))

