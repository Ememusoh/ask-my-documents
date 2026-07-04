"""Grounded prompt construction using retrieved document evidence.

This module does not call an AI model. It only builds the instructions and
source passages that the generator will send to the model later.
"""

from __future__ import annotations

from typing import Any


RetrievedChunk = dict[str, Any]
ChatMessage = dict[str, str]


ABSTAIN_MESSAGE = (
    "I do not have enough information in the uploaded documents to answer that."
)


def build_grounded_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """Build a prompt that forces answers to stay grounded in retrieved sources."""
    if not question.strip():
        raise ValueError("question cannot be empty.")

    sources = format_sources(retrieved_chunks)

    return f"""You are Ask My Documents, a document question-answering assistant.

Answer the question using only the source passages below.
If the sources do not contain the answer, say exactly:
"{ABSTAIN_MESSAGE}"

Rules:
- Do not use outside knowledge.
- Cite sources using labels like [Source 1].
- Keep the answer clear and concise.

Question:
{question}

Source passages:
{sources}

Answer:"""


def format_sources(retrieved_chunks: list[RetrievedChunk]) -> str:
    """Format only source labels and passage text for the prompt.

    Full metadata stays outside the prompt so we can display it in the app without
    paying for extra input tokens in the OpenAI request.
    """
    if not retrieved_chunks:
        return "No relevant source passages were retrieved."

    formatted_sources = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        text = str(chunk.get("text", "")).strip()

        # Keep a tiny source label so the model can cite evidence as [Source 1].
        formatted_sources.append(f"[Source {index}]\n{text}")

    return "\n\n".join(formatted_sources)


def build_chat_messages(
    question: str, retrieved_chunks: list[RetrievedChunk]
) -> list[ChatMessage]:
    """Wrap the grounded prompt in a simple chat-message format for the generator."""
    return [
        {
            "role": "system",
            "content": "You answer questions only from the provided document sources.",
        },
        {
            "role": "user",
            "content": build_grounded_prompt(question, retrieved_chunks),
        },
    ]
