"""Tests for simple evaluation metrics."""

from rag.evaluator import (
    build_chunk_detail_records,
    check_abstention_correctness,
    check_answer_correctness,
    check_citation_correctness,
    check_retrieval_hit,
    evaluate_single_result,
    run_evaluation,
    save_chunk_detail_records,
    summarize_results,
)


def test_retrieval_hit_checks_expected_source() -> None:
    """Retrieval hit should be true when the expected source was retrieved."""
    chunks = [{"citation": "Project_Proposal.docx, Paragraph 3"}]

    assert check_retrieval_hit(chunks, "Paragraph 3") is True
    assert check_retrieval_hit(chunks, "Paragraph 9") is False


def test_answer_and_citation_correctness() -> None:
    """Answer and citation checks should use simple visible text rules."""
    answer = "The system is a RAG prototype. [Source 1]"
    sources = [{"source_number": 1}]

    assert check_answer_correctness(answer, "RAG prototype") is True
    assert check_citation_correctness(answer, sources) is True


def test_abstention_correctness() -> None:
    """Abstention should be correct when expected and incorrect otherwise."""
    assert check_abstention_correctness("No answer", should_abstain=True, abstained=True)
    assert not check_abstention_correctness(
        "No answer", should_abstain=False, abstained=True
    )


def test_evaluate_single_result_and_summary() -> None:
    """A full evaluation row should produce clear boolean metrics."""
    question = {
        "question_id": "q1",
        "question": "What is the project?",
        "expected_answer": "RAG prototype",
        "expected_source": "Paragraph 3",
        "is_answerable": True,
    }
    retrieval_response = {
        "chunks": [{"citation": "Project_Proposal.docx, Paragraph 3"}]
    }
    generated_answer = {
        "answer": "The project is a RAG prototype. [Source 1]",
        "sources": [{"source_number": 1}],
        "abstained": False,
    }

    record = evaluate_single_result(
        question,
        retrieval_response,
        generated_answer,
        response_time_seconds=1.23456,
        configuration_name="fixed_top_3",
    )
    summary = summarize_results([record])

    assert record["retrieval_hit"] is True
    assert record["answer_correct"] is True
    assert record["citation_correct"] is True
    assert summary["answer_correct_rate"] == 1.0


def test_build_chunk_detail_records_marks_llm_sources() -> None:
    """Chunk detail rows should show what was retrieved and what reached the LLM."""
    question = {
        "question_id": "q1",
        "question": "What is the project aim?",
    }
    retrieval_response = {
        "namespace": "fixed_chunks",
        "top_k": 3,
        "relevance_threshold": 0.35,
        "raw_chunks": [
            {
                "id": "chunk-1",
                "score": 0.52,
                "citation": "Project_Proposal.docx, paragraphs 1–2",
                "document_name": "Project_Proposal.docx",
                "source_label": "paragraphs 1–2",
                "chunking_strategy": "fixed_size",
                "chunk_number": 1,
                "text": "Useful project aim text.",
            },
            {
                "id": "chunk-2",
                "score": 0.21,
                "citation": "Project_Proposal.docx, paragraphs 7–8",
                "document_name": "Project_Proposal.docx",
                "source_label": "paragraphs 7–8",
                "chunking_strategy": "fixed_size",
                "chunk_number": 2,
                "text": "Weak match text.",
            },
        ],
        "chunks": [
            {
                "id": "chunk-1",
                "score": 0.52,
                "citation": "Project_Proposal.docx, paragraphs 1–2",
                "text": "Useful project aim text.",
            }
        ],
    }

    records = build_chunk_detail_records(
        question, retrieval_response, configuration_name="fixed_top_3"
    )

    assert len(records) == 2
    assert list(records[0].keys()) == [
        "configuration",
        "question_id",
        "question",
        "threshold",
        "pinecone_rank",
        "pinecone_score",
        "sent_to_llm",
        "llm_source_number",
        "source",
        "chunk_text",
    ]
    assert records[0]["sent_to_llm"] is True
    assert records[0]["llm_source_number"] == 1
    assert records[0]["pinecone_rank"] == 1
    assert records[0]["pinecone_score"] == 0.52
    assert records[0]["source"] == "Project_Proposal.docx, paragraphs 1–2"
    assert records[1]["sent_to_llm"] is False
    assert records[1]["llm_source_number"] == ""


def test_save_chunk_detail_records_writes_csv(tmp_path) -> None:
    """Chunk detail records should be saved as a normal CSV file."""
    records = [
        {
            "configuration": "fixed_top_3",
            "question_id": "q1",
            "sent_to_llm": True,
            "chunk_text": "Example chunk text.",
        }
    ]
    output_path = tmp_path / "chunk_details.csv"

    save_chunk_detail_records(records, output_path)

    saved_text = output_path.read_text(encoding="utf-8")
    assert "configuration,question_id,sent_to_llm,chunk_text" in saved_text
    assert "fixed_top_3,q1,True,Example chunk text." in saved_text


def test_run_evaluation_saves_results_without_real_api_calls(tmp_path) -> None:
    """The runner should create both result CSVs using fake test functions."""
    questions_path = tmp_path / "eval_questions.csv"
    results_path = tmp_path / "evaluation_results.csv"
    chunk_details_path = tmp_path / "evaluation_chunk_details.csv"

    questions_path.write_text(
        "\n".join(
            [
                "question_id,question,expected_answer,expected_source,is_answerable",
                "q1,What is the project?,RAG prototype,paragraph 3,True",
                "q2,Who founded America?,,,False",
            ]
        ),
        encoding="utf-8",
    )

    def fake_retrieve(question, namespace, top_k, relevance_threshold):
        strong_chunk = {
            "id": f"{namespace}-{top_k}-strong",
            "score": 0.8,
            "citation": "Project_Proposal.docx, paragraph 3",
            "document_name": "Project_Proposal.docx",
            "source_label": "paragraph 3",
            "chunking_strategy": "fixed_size",
            "chunk_number": 1,
            "text": "This project is a RAG prototype.",
        }
        weak_chunk = {
            "id": f"{namespace}-{top_k}-weak",
            "score": 0.1,
            "citation": "Project_Proposal.docx, paragraph 9",
            "document_name": "Project_Proposal.docx",
            "source_label": "paragraph 9",
            "chunking_strategy": "fixed_size",
            "chunk_number": 2,
            "text": "Weak unrelated text.",
        }

        if "Who founded America" in question:
            raw_chunks = [weak_chunk]
            sent_chunks = []
        else:
            raw_chunks = [strong_chunk, weak_chunk]
            sent_chunks = [strong_chunk]

        return {
            "namespace": namespace,
            "top_k": top_k,
            "relevance_threshold": relevance_threshold,
            "raw_retrieved_count": len(raw_chunks),
            "filtered_count": len(sent_chunks),
            "raw_chunks": raw_chunks,
            "chunks": sent_chunks,
        }

    def fake_generate(question, retrieved_chunks, max_output_tokens):
        return {
            "answer": "The project is a RAG prototype. [Source 1]",
            "sources": [{"source_number": 1, "citation": retrieved_chunks[0]["citation"]}],
            "abstained": False,
        }

    report = run_evaluation(
        questions_csv_path=questions_path,
        results_output_path=results_path,
        chunk_details_output_path=chunk_details_path,
        retrieve_function=fake_retrieve,
        generate_function=fake_generate,
    )

    assert len(report["results"]) == 8
    assert len(report["chunk_details"]) == 12
    assert results_path.exists()
    assert chunk_details_path.exists()
    assert "sent_to_llm" in chunk_details_path.read_text(encoding="utf-8")
