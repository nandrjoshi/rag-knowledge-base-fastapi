from fastapi import FastAPI
from rag_knowledge_base_fastapi.config.settings import settings
from rag_knowledge_base_fastapi.services.db import create_db_engine, check_db_health

from rag_knowledge_base_fastapi.models.ingest import IngestTextRequest
from rag_knowledge_base_fastapi.services.chunking import chunk_text
from rag_knowledge_base_fastapi.services.kb_repository import insert_chunks_with_embeddings
from rag_knowledge_base_fastapi.config.settings import settings


app = FastAPI(title ="RAG Knowledge Base (FastAPI)", version="0.1.0")

@app.get("/health")
def health() -> dict:
    return {"status":"ok"}

@app.get("/config")
def config() -> dict:
    return {
        "openai_chat_model": settings.openai_chat_model,
        "openai_embed_model": settings.openai_embed_model,
        "database_url": settings.database_url,
        "rag_top_k_default": settings.rag_top_k_default,
        "chunk_size_chars": settings.chunk_size_chars,
        "chunk_overlap_chars": settings.chunk_overlap_chars,
        "openai_api_key_configured": bool(settings.openai_api_key),
    }

@app.get("/db/health")
def db_health() -> dict:
    engine = create_db_engine()
    result = check_db_health(engine)
    return {
        "ok": result.ok,
        "server_version": result.server_version,
        "error": result.error,
    }

@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest) -> dict:
    chunks = chunk_text(
        req.content,
        chunk_size=settings.chunk_size_chars,
        chunk_overlap=settings.chunk_overlap_chars,
    )

    result = insert_chunks_with_embeddings(
        source=req.source,
        doc_id=req.doc_id,
        chunks=[(c.chunk_index, c.content) for c in chunks],
        metadata={"ingest_type": "text"},
    )

    return {
        "source": req.source,
        "doc_id": req.doc_id,
        "chunks_created": len(chunks),
        "chunks_inserted": result.inserted,
    }
