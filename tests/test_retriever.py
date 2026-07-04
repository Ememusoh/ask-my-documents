"""Tests for retrieval filtering and abstention logic."""

from rag import retriever


def test_retrieval_returns_highest_scoring_chunks_first(monkeypatch) -> None:
    """Retriever should keep relevant chunks and order strongest evidence first."""

    def fake_search_chunks(*args, **kwargs):
        return [
            {"score": 0.42, "chunk_text": "weak", "document_name": "demo.txt"},
            {"score": 0.91, "chunk_text": "strong", "document_name": "demo.txt"},
            {"score": 0.70, "chunk_text": "good", "document_name": "demo.txt"},
        ]

    monkeypatch.setattr(retriever, "search_chunks", fake_search_chunks)

    response = retriever.retrieve_relevant_chunks(
        question="What matters here?",
        namespace="fixed_chunks",
        relevance_threshold=0.5,
    )

    assert response["can_answer"] is True
    assert [chunk["score"] for chunk in response["chunks"]] == [0.91, 0.70]
    assert [chunk["score"] for chunk in response["raw_chunks"]] == [0.42, 0.91, 0.70]
    assert response["raw_retrieved_count"] == 3
    assert response["filtered_count"] == 2


def test_relevance_threshold_supports_abstention(monkeypatch) -> None:
    """Retriever should abstain when no chunk passes the threshold."""

    def fake_search_chunks(*args, **kwargs):
        return [{"score": 0.20, "chunk_text": "not relevant"}]

    monkeypatch.setattr(retriever, "search_chunks", fake_search_chunks)

    response = retriever.retrieve_relevant_chunks(
        question="Question not answered by the document",
        namespace="fixed_chunks",
        relevance_threshold=0.5,
    )

    assert response["can_answer"] is False
    assert response["chunks"] == []
    assert len(response["raw_chunks"]) == 1


def test_filter_prepared_chunks_reuses_raw_retrieval_results() -> None:
    """Changing threshold should filter existing chunks without a new search."""
    chunks = [
        {"score": 0.20, "text": "weak"},
        {"score": 0.50, "text": "medium"},
        {"score": 0.80, "text": "strong"},
    ]

    filtered_chunks = retriever.filter_prepared_chunks(chunks, relevance_threshold=0.45)

    assert [chunk["text"] for chunk in filtered_chunks] == ["strong", "medium"]


def test_citation_groups_repeated_paragraph_labels() -> None:
    """Citation display should not repeat the same document name many times."""
    result = {
        "document_name": "Project_Proposal.docx",
        "source_labels": [
            "Project_Proposal.docx, paragraph 9",
            "Project_Proposal.docx, paragraph 10",
            "Project_Proposal.docx, paragraph 11",
            "Project_Proposal.docx, paragraph 13",
        ],
    }

    citation = retriever.make_citation(result)

    assert citation == "Project_Proposal.docx, paragraphs 9-11, 13"
