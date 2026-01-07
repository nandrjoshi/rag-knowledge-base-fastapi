from __future__ import annotations
from dataclasses import dataclass
from openai import OpenAI
from rag_knowledge_base_fastapi.config.settings import settings

@dataclass
class EmbeddingResult:
    model: str
    vector: list[float]

class OpenAIEmbeddingsClient:
    """
    Minimal OpenAI embeddings client wrapper.

    Uses:
    - OPENAI_API_KEY
    - OPENAI_EMBED_MODEL
    from settings.
    """
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embed_model

    @property
    def model(self) -> str:
        return self._model

    def embed_text(self, text: str) -> EmbeddingResult:
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot embed empty text.")

        resp = self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        vector = resp.data[0].embedding
        return EmbeddingResult(model=self._model, vector=vector)