"""Simple retrieval logic for Ask My Documents.

Pinecone does the vector search and cosine-similarity scoring. This module adds
the project-specific retrieval rules: top-k choice, relevance threshold checking,
abstention, and citation-friendly output.
"""

from __future__ import annotations

import re
from typing import Any

from rag.embeddings import search_chunks


SearchResult = dict[str, Any]
RetrievalResponse = dict[str, Any]


def retrieve_relevant_chunks(
    question: str,
    namespace: str,
    top_k: int = 3,
    relevance_threshold: float = 0.5,
    text_field: str = "chunk_text",
) -> RetrievalResponse:
    """Search Pinecone, filter weak matches, and return clean retrieval results."""
    if not question.strip():
        raise ValueError("question cannot be empty.")

    # Pinecone embeds the question, compares it to stored chunks, and returns matches.
    search_results = search_chunks(
        question=question,
        namespace=namespace,
        top_k=top_k,
        text_field=text_field,
    )

    raw_chunks = [
        prepare_retrieved_chunk(result, text_field=text_field)
        for result in search_results
    ]
    filtered_chunks = filter_prepared_chunks(raw_chunks, relevance_threshold)

    return {
        "can_answer": bool(filtered_chunks),
        "message": _retrieval_message(bool(filtered_chunks)),
        "question": question,
        "namespace": namespace,
        "top_k": top_k,
        "relevance_threshold": relevance_threshold,
        "raw_retrieved_count": len(raw_chunks),
        "filtered_count": len(filtered_chunks),
        "retrieved_count": len(filtered_chunks),
        "raw_chunks": raw_chunks,
        "chunks": filtered_chunks,
    }


def filter_by_relevance(
    search_results: list[SearchResult], relevance_threshold: float
) -> list[SearchResult]:
    """Keep only chunks whose Pinecone similarity score is high enough."""
    _validate_relevance_threshold(relevance_threshold)

    relevant_results = [
        result
        for result in search_results
        if passes_relevance_threshold(_score(result), relevance_threshold)
    ]

    # Keep the strongest evidence first for the prompt and citation display.
    return sorted(relevant_results, key=_score, reverse=True)


def passes_relevance_threshold(score: float, relevance_threshold: float) -> bool:
    """Return True when a similarity score is strong enough to use as evidence."""
    _validate_relevance_threshold(relevance_threshold)
    return score >= relevance_threshold


def filter_prepared_chunks(
    chunks: list[SearchResult], relevance_threshold: float
) -> list[SearchResult]:
    """Filter already-retrieved chunks without calling Pinecone again."""
    _validate_relevance_threshold(relevance_threshold)
    relevant_chunks = [
        chunk
        for chunk in chunks
        if passes_relevance_threshold(_score(chunk), relevance_threshold)
    ]
    return sorted(relevant_chunks, key=_score, reverse=True)


def prepare_retrieved_chunk(
    result: SearchResult, text_field: str = "chunk_text"
) -> SearchResult:
    """Return one retrieved chunk in a clean format for prompts and citations."""
    return {
        "id": result.get("id"),
        "text": result.get(text_field, ""),
        "score": _score(result),
        "citation": make_citation(result),
        "document_name": result.get("document_name", "Unknown document"),
        "source_label": result.get("source_label", "Unknown source"),
        "chunking_strategy": result.get("chunking_strategy"),
        "chunk_number": result.get("chunk_number"),
        "word_count": result.get("word_count"),
    }


def make_citation(result: SearchResult) -> str:
    """Create a short citation label from the chunk metadata."""
    document_name = result.get("document_name", "Unknown document")
    source_labels = result.get("source_labels")
    if isinstance(source_labels, list) and source_labels:
        return _compact_source_labels(str(document_name), source_labels)

    source_label = result.get("source_label", "Unknown source")
    if str(source_label).startswith(str(document_name)):
        return str(source_label)
    return f"{document_name}, {source_label}"


def _compact_source_labels(document_name: str, source_labels: list[str]) -> str:
    """Group repeated document labels into a shorter citation."""
    paragraph_numbers = _source_numbers(source_labels, "paragraph")
    if paragraph_numbers:
        return f"{document_name}, paragraphs {_format_number_ranges(paragraph_numbers)}"

    page_numbers = _source_numbers(source_labels, "page")
    if page_numbers:
        return f"{document_name}, pages {_format_number_ranges(page_numbers)}"

    unique_labels = list(dict.fromkeys(str(label) for label in source_labels))
    return " | ".join(unique_labels)


def _source_numbers(source_labels: list[str], source_type: str) -> list[int]:
    """Extract paragraph or page numbers from source labels."""
    numbers = []
    pattern = rf"{source_type} (\d+)"

    for label in source_labels:
        match = re.search(pattern, str(label), flags=re.IGNORECASE)
        if match:
            numbers.append(int(match.group(1)))

    return sorted(set(numbers))


def _format_number_ranges(numbers: list[int]) -> str:
    """Turn [9, 10, 11, 13] into '9-11, 13'."""
    ranges = []
    start = previous = numbers[0]

    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue

        ranges.append(_format_one_range(start, previous))
        start = previous = number

    ranges.append(_format_one_range(start, previous))
    return ", ".join(ranges)


def _format_one_range(start: int, end: int) -> str:
    """Format one number or one continuous number range."""
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _score(result: SearchResult) -> float:
    """Read a Pinecone score safely as a float."""
    try:
        return float(result.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _validate_relevance_threshold(relevance_threshold: float) -> None:
    """Keep the threshold easy to explain as a value from 0 to 1."""
    if not 0 <= relevance_threshold <= 1:
        raise ValueError("relevance_threshold must be between 0 and 1.")


def _retrieval_message(can_answer: bool) -> str:
    """Explain whether retrieval found enough evidence."""
    if can_answer:
        return "Relevant chunks were found."
    return "The uploaded documents do not contain enough information to answer."
