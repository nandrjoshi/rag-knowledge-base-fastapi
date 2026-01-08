# RAG Knowledge Base (FastAPI + pgvector + OpenAI)

A simple Retrieval-Augmented Generation (RAG) knowledge assistant built with:
- FastAPI (API layer)
- PostgreSQL + pgvector (vector store)
- OpenAI embeddings + chat completion (LLM)
- Poetry (dependency management)

## Features
- Ingest text: chunk → embed → store in Postgres (pgvector)
- Vector search: semantic retrieval using pgvector similarity
- Chat: RAG prompting with citations from retrieved chunks
- Minimal chatbot UI served by FastAPI

## Architecture (high level)
1. **Ingestion**
   - Input text is split into overlapping chunks
   - Each chunk is embedded using OpenAI embeddings
   - Chunks + metadata + vectors are stored in `kb_chunks`

2. **Retrieval**
   - User query is embedded
   - pgvector similarity search retrieves top-k chunks
   - Chunks are returned via `/search`

3. **Chat (RAG)**
   - Retrieved chunks are injected as context
   - LLM answers using only context and provides citations
   - Response returned via `/chat`

## API Endpoints
- `POST /ingest/text` — ingest raw text into the knowledge base
- `POST /search` — retrieve relevant chunks via vector similarity
- `POST /chat` — answer questions using RAG + citations
- `GET /` — minimal chatbot UI

## Configuration
Environment variables (use `.env`, do not commit):
- `OPENAI_API_KEY`
- `OPENAI_EMBED_MODEL` (default: text-embedding-3-small)
- `OPENAI_CHAT_MODEL` (example: gpt-4o-mini)
- `DATABASE_URL` (Postgres connection string)

## Local Run
1. Start Postgres with pgvector enabled
2. Install dependencies:
   - `poetry install`
3. Run API:
   - `poetry run uvicorn rag_knowledge_base_fastapi.main:app --reload`
4. Open UI:
   - http://127.0.0.1:8000/
