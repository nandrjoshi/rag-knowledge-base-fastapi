from __future__ import annotations

from sqlalchemy import text

from rag_knowledge_base_fastapi.services.db import create_db_engine


# NOTE: text-embedding-3-small produces 1536-dim vectors.
# We'll make this configurable later once the embedding pipeline is implemented.
EMBED_DIM = 1536


def init_db() -> None:
    """
    Initializes pgvector extension + creates the embeddings table (idempotent).
    Safe to run multiple times.
    """
    engine = create_db_engine()

    ddl = f"""
    -- Enable pgvector
    CREATE EXTENSION IF NOT EXISTS vector;

    -- Store chunked documents with embeddings for similarity search
    CREATE TABLE IF NOT EXISTS kb_chunks (
        id BIGSERIAL PRIMARY KEY,
        source TEXT NOT NULL,               -- e.g., filename/url
        doc_id TEXT NULL,                   -- logical document id (optional)
        chunk_index INT NOT NULL DEFAULT 0, -- chunk number within doc
        content TEXT NOT NULL,              -- chunk text
        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        embedding VECTOR({EMBED_DIM}) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    -- Helpful indexes (non-vector)
    CREATE INDEX IF NOT EXISTS kb_chunks_source_idx ON kb_chunks (source);
    CREATE INDEX IF NOT EXISTS kb_chunks_doc_id_idx ON kb_chunks (doc_id);

    -- Vector index will be added after we ingest data (best practice).
    -- (We will choose HNSW vs IVFFLAT based on your installed pgvector version and dataset size.)
    """

    with engine.begin() as conn:
        # Execute each statement safely (split on ';' but ignore empties)
        for stmt in (s.strip() for s in ddl.split(";")):
            if stmt:
                conn.execute(text(stmt))


if __name__ == "__main__":
    init_db()
    print("DB initialized: pgvector extension + kb_chunks table ensured.")
