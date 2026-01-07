from fastapi import FastAPI
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