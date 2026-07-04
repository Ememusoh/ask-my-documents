"""Answer generation constrained to retrieved evidence.

This module calls OpenAI only after retrieval has found relevant chunks. The
prompt receives source labels and passage text, while full source metadata is
returned separately for display.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from rag.prompts import ABSTAIN_MESSAGE, build_grounded_prompt


RetrievedChunk = dict[str, Any]
GeneratedAnswer = dict[str, Any]


def generate_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    model: str | None = None,
    max_output_tokens: int = 300,
    reasoning_effort: str = "low",
    client: Any | None = None,
) -> GeneratedAnswer:
    """Generate one grounded answer using only the retrieved chunks."""
    if not retrieved_chunks:
        return build_abstention_response()

    # The prompt contains only source labels and passage text to reduce token cost.
    prompt = build_grounded_prompt(question, retrieved_chunks)
    openai_client = client or get_openai_client()
    model = model or get_chat_model()

    response = openai_client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
    )
    answer = str(response.output_text).strip()
    if not answer:
        raise ValueError("OpenAI returned no visible answer. Try a larger token limit.")

    return {
        "answer": answer,
        "sources": build_source_display(retrieved_chunks),
        "abstained": answer == ABSTAIN_MESSAGE,
    }


def build_abstention_response() -> GeneratedAnswer:
    """Return the standard response when retrieval finds no useful evidence."""
    return {
        "answer": ABSTAIN_MESSAGE,
        "sources": [],
        "abstained": True,
    }


def build_source_display(retrieved_chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Keep source metadata outside the prompt so the app can display it later."""
    sources = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        sources.append(
            {
                "source_number": index,
                "label": f"Source {index}",
                "citation": chunk.get("citation", "Unknown source"),
                "document_name": chunk.get("document_name", "Unknown document"),
                "source_label": chunk.get("source_label", "Unknown source"),
                "score": chunk.get("score"),
                "text": chunk.get("text", ""),
            }
        )

    return sources


def get_openai_client() -> OpenAI:
    """Create an OpenAI client using the API key from .env."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key or "your_" in api_key:
        raise ValueError("Add your real OPENAI_API_KEY to the .env file.")

    return OpenAI(api_key=api_key)


def get_chat_model() -> str:
    """Read the OpenAI chat model name from .env."""
    load_dotenv()
    model = os.getenv("OPENAI_CHAT_MODEL", "").strip()

    if not model or "your_" in model:
        raise ValueError("Add OPENAI_CHAT_MODEL to the .env file.")

    return model
