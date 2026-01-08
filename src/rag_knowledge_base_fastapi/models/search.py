from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="User query to search against the knowledge base")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    doc_id: str | None = Field(default=None, description="Optional filter to a specific document id")
    source: str | None = Field(default=None, description="Optional filter to a specific source")


class SearchHit(BaseModel):
    id: int
    source: str
    doc_id: str | None
    chunk_index: int
    content: str
    score: float = Field(..., description="Similarity score (lower distance => better match)")


class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: list[SearchHit]
