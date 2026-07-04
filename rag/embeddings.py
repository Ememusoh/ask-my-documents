"""Pinecone indexing helpers for Ask My Documents.

Pinecone is configured to do integrated embedding for this project. That means
we send chunk text to Pinecone, and Pinecone turns that text into vectors and
stores the vectors for search.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone


Chunk = dict[str, Any]
Record = dict[str, Any]


METADATA_KEYS = [
    "document_name",
    "file_type",
    "source_id",
    "source_type",
    "source_label",
    "chunk_id",
    "chunk_number",
    "chunking_strategy",
    "chunk_size_words",
    "overlap_words",
    "word_count",
    "source_ids",
    "source_labels",
]


def get_pinecone_index() -> Any:
    """Connect to the Pinecone index using values from the local .env file."""
    # Load secrets/settings from .env so keys are not hardcoded in the code.
    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    index_host = os.getenv("PINECONE_INDEX_HOST")

    if not api_key or "your_" in api_key:
        raise ValueError("Add your real PINECONE_API_KEY to the .env file.")
    if not index_host or "your_" in index_host:
        raise ValueError("Add your real PINECONE_INDEX_HOST to the .env file.")

    pinecone_client = Pinecone(api_key=api_key)
    return pinecone_client.Index(host=index_host)


def chunk_to_record(chunk: Chunk, text_field: str = "chunk_text") -> Record:
    """Convert one project chunk into one Pinecone record.

    Pinecone needs two main things:
    1. ``_id``: a unique ID for the chunk.
    2. ``chunk_text``: the text Pinecone should embed.

    The remaining fields are metadata used later for citations and evaluation.
    """
    chunk_text = str(chunk.get("text", "")).strip()
    if not chunk_text:
        raise ValueError("Cannot index a chunk with empty text.")

    record = {
        "_id": str(chunk.get("chunk_id", f"chunk-{chunk.get('chunk_number', 0)}")),
        text_field: chunk_text,
    }

    # Keep only simple metadata fields that help us explain sources later.
    for key in METADATA_KEYS:
        if key in chunk and chunk[key] is not None:
            record[key] = chunk[key]

    return record


def chunks_to_records(chunks: list[Chunk], text_field: str = "chunk_text") -> list[Record]:
    """Convert many chunks into Pinecone records."""
    return [chunk_to_record(chunk, text_field=text_field) for chunk in chunks]


def index_chunks(
    chunks: list[Chunk],
    namespace: str,
    batch_size: int = 50,
    text_field: str | None = None,
) -> int:
    """Upload chunks to Pinecone and let Pinecone embed them automatically."""
    if not chunks:
        return 0

    load_dotenv()
    text_field = text_field or os.getenv("PINECONE_TEXT_FIELD", "chunk_text")

    index = get_pinecone_index()
    records = chunks_to_records(chunks, text_field=text_field)

    # Send records in small batches so large documents do not create one huge request.
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        index.upsert_records(records=batch, namespace=namespace)

    return len(records)


def delete_document_chunks(document_name: str, namespace: str) -> None:
    """Delete existing chunks for one document before re-indexing it.

    If the namespace does not exist yet, there is nothing to delete. This can
    happen the first time we index a new chunking strategy.
    """
    index = get_pinecone_index()
    try:
        index.delete(
            namespace=namespace,
            filter={"document_name": {"$eq": document_name}},
        )
    except Exception as error:
        if _is_missing_namespace_error(error):
            return
        raise


def search_chunks(
    question: str,
    namespace: str,
    top_k: int = 3,
    text_field: str | None = None,
) -> list[Record]:
    """Search Pinecone with a question and return the most relevant chunks."""
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive whole number.")

    load_dotenv()
    text_field = text_field or os.getenv("PINECONE_TEXT_FIELD", "chunk_text")

    index = get_pinecone_index()
    response = index.search(
        namespace=namespace,
        inputs={"text": question},
        top_k=top_k,
        fields=[text_field, *METADATA_KEYS],
    )

    return _search_response_to_hits(response)


def _search_response_to_hits(response: Any) -> list[Record]:
    """Turn Pinecone's search response into a simple list of result dictionaries."""
    if hasattr(response, "to_dict"):
        response = response.to_dict()

    hits = response.get("result", {}).get("hits", [])
    results: list[Record] = []

    for hit in hits:
        fields = hit.get("fields", {})
        results.append(
            {
                "id": _first_present(hit, ["id", "id_", "_id"]),
                "score": _first_present(hit, ["score", "score_", "_score"]),
                **fields,
            }
        )

    return results


def _first_present(data: dict[str, Any], keys: list[str]) -> Any:
    """Return the first available value from possible Pinecone field names."""
    for key in keys:
        if key in data:
            return data[key]
    return None


def _is_missing_namespace_error(error: Exception) -> bool:
    """Check whether Pinecone is saying this namespace has not been created yet."""
    error_text = str(error).lower()
    return "namespace not found" in error_text or "404" in error_text
