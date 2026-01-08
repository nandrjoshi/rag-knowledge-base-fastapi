from fastapi import FastAPI
from rag_knowledge_base_fastapi.config.settings import settings
from rag_knowledge_base_fastapi.services.db import create_db_engine, check_db_health

from rag_knowledge_base_fastapi.models.ingest import IngestTextRequest
from rag_knowledge_base_fastapi.services.chunking import chunk_text
from rag_knowledge_base_fastapi.services.kb_repository import insert_chunks_with_embeddings
from rag_knowledge_base_fastapi.models.search import SearchRequest, SearchResponse, SearchHit
from rag_knowledge_base_fastapi.services.retrieval import search_chunks

from rag_knowledge_base_fastapi.models.chat import ChatRequest, ChatResponse, Citation
from rag_knowledge_base_fastapi.services.chat_service import answer_with_rag

from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles



app = FastAPI(title ="RAG Knowledge Base (FastAPI)", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    hits = search_chunks(
        query=req.query,
        top_k=req.top_k,
        doc_id=req.doc_id,
        source=req.source,
    )

    return SearchResponse(
        query=req.query,
        top_k=req.top_k,
        hits=[
            SearchHit(
                id=h.id,
                source=h.source,
                doc_id=h.doc_id,
                chunk_index=h.chunk_index,
                content=h.content,
                score=h.score,
            )
            for h in hits
        ],
    )

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = answer_with_rag(
        message=req.message,
        top_k=req.top_k,
        doc_id=req.doc_id,
        source=req.source,
    )

    return ChatResponse(
        message=req.message,
        answer=result.answer,
        citations=result.citations,
    )

@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    html = (STATIC_DIR / "chat.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)
