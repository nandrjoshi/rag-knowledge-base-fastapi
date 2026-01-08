from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message/question")
    top_k: int = Field(default=5, ge=1, le=20, description="How many chunks to retrieve for context")
    doc_id: str | None = Field(default=None, description="Optional filter to a specific document id")
    source: str | None = Field(default=None, description="Optional filter to a specific source")


class Citation(BaseModel):
    id: int
    source: str
    doc_id: str | None
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    message: str
    answer: str
    citations: list[Citation]
