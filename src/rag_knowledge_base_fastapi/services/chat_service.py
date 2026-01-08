from __future__ import annotations

import re

from dataclasses import dataclass

from openai import OpenAI

from rag_knowledge_base_fastapi.config.settings import settings
from rag_knowledge_base_fastapi.models.chat import Citation
from rag_knowledge_base_fastapi.services.retrieval import search_chunks


@dataclass(frozen=True)
class ChatResult:
    answer: str
    citations: list[Citation]


def _build_context(hits) -> str:
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        # Keep content as-is; we can add truncation later if needed.
        lines.append(f"[{i}] source={h.source} doc_id={h.doc_id} chunk={h.chunk_index} score={h.score:.4f}\n{h.content}")
    return "\n\n".join(lines)


def answer_with_rag(
    *,
    message: str,
    top_k: int = 5,
    doc_id: str | None = None,
    source: str | None = None,
) -> ChatResult:
    message = (message or "").strip()
    if not message:
        raise ValueError("message is required")

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    hits = search_chunks(query=message, top_k=top_k, doc_id=doc_id, source=source)

    context = _build_context(hits)

    system_prompt = (
        "You are a helpful knowledge-base assistant.\n"
        "Answer the user's question using ONLY the provided context.\n"
        "If the context is insufficient, say you don't know and ask a clarifying question.\n"
        "When you use information from a chunk, cite it like [1], [2], etc.\n"
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"User question:\n{message}\n\n"
        f"Answer:"
    )

    client = OpenAI(api_key=settings.openai_api_key)

    # If you don't have chat model in settings yet, default here.
    chat_model = getattr(settings, "openai_chat_model", None) or "gpt-4o-mini"

    resp = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content or ""

    cited = _extract_cited_indices(answer)

    # citations = [
    #     Citation(
    #         id=h.id,
    #         source=h.source,
    #         doc_id=h.doc_id,
    #         chunk_index=h.chunk_index,
    #         score=h.score,
    #     )
    #     for h in hits
    # ]

    citations = []
    for i, h in enumerate(hits, start=1):
        if i in cited:
            citations.append(
                Citation(
                    id=h.id,
                    source=h.source,
                    doc_id=h.doc_id,
                    chunk_index=h.chunk_index,
                    score=h.score,
                )
            )

    return ChatResult(answer=answer.strip(), citations=citations)

def _extract_cited_indices(answer: str) -> set[int]:
        # Matches [1], [2], etc.
        found = re.findall(r"\[(\d+)\]", answer or "")
        return {int(x) for x in found if x.isdigit()}
