from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy import text

from rag_knowledge_base_fastapi.services.db import create_db_engine
from rag_knowledge_base_fastapi.services.openai_embeddings import OpenAIEmbeddingsClient


@dataclass(frozen=True)
class RetrievalHit:
    id: int
    source: str
    doc_id: str | None
    chunk_index: int
    content: str
    score: float  # pgvector distance (lower is better)


def _vec_literal(vec: list[float]) -> str:
    # pgvector expects: '[0.1,0.2,...]'
    return "[" + ",".join(map(str, vec)) + "]"


def search_chunks(
    *,
    query: str,
    top_k: int = 5,
    doc_id: str | None = None,
    source: str | None = None,
) -> list[RetrievalHit]:
    query = (query or "").strip()
    if not query:
        raise ValueError("query is required")
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")

    embedder = OpenAIEmbeddingsClient()
    q = embedder.embed_text(query)
    qvec = _vec_literal(q.vector)

    engine = create_db_engine()

    # Note: <-> is Euclidean distance for vector type; lower is better.
    # We'll keep it as "score" for now.
    base_sql = """
        SELECT
            id,
            source,
            doc_id,
            chunk_index,
            content,
            (embedding <-> CAST(:qvec AS vector)) AS score
        FROM kb_chunks
        WHERE 1=1
    """

    params: dict[str, object] = {"qvec": qvec, "limit": top_k}

    if doc_id:
        base_sql += " AND doc_id = :doc_id"
        params["doc_id"] = doc_id

    if source:
        base_sql += " AND source = :source"
        params["source"] = source

    base_sql += " ORDER BY embedding <-> CAST(:qvec AS vector) LIMIT :limit"

    sql = text(base_sql)

    hits: list[RetrievalHit] = []
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    for r in rows:
        hits.append(
            RetrievalHit(
                id=int(r[0]),
                source=str(r[1]),
                doc_id=r[2],
                chunk_index=int(r[3]),
                content=str(r[4]),
                score=float(r[5]),
            )
        )

    return hits
