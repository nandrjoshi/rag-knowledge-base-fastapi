from __future__ import annotations

from pydantic import BaseModel, Field

class IngestTextRequest(BaseModel):
    source: str = Field(..., description="Logical source identifier,  e.g. filename or URL")
    content: str = Field(..., description="Raw text content to chunk and ingest")
    doc_id: str | None = Field(default=None, description="Optional document id for grouping")
    