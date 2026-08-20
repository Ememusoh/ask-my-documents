"""Evaluation helpers for Ask My Documents.

The evaluator checks results that were already produced by retrieval and
generation. This keeps the metrics simple, cheap to test, and easy to explain.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from rag.prompts import ABSTAIN_MESSAGE


EvaluationQuestion = dict[str, Any]
EvaluationRecord = dict[str, Any]

REQUIRED_COLUMNS = [
    "question_id",
    "question",
    "expected_answer",
    "expected_source",
    "is_answerable",
]

DEFAULT_QUESTIONS_PATH = Path("data/evaluation/eval_questions.csv")
DEFAULT_RESULTS_PATH = Path("data/results/evaluation_results.csv")
DEFAULT_CHUNK_DETAILS_PATH = Path("data/results/evaluation_chunk_details.csv")
DEFAULT_RELEVANCE_THRESHOLD = 0.35
DEFAULT_MAX_OUTPUT_TOKENS = 1000

RESULT_FIELDNAMES = [
    "run_id",
    "configuration",
    "chunking_strategy",
    "namespace",
    "top_k",
    "relevance_threshold",
    "question_id",
    "question",
    "is_answerable",
    "expected_answer",
    "expected_source",
    "raw_retrieved_count",
    "chunks_sent_to_llm",
    "retrieval_hit",
    "expected_source_rank",
    "expected_source_score",
    "expected_source_filtered_out",
    "generated_answer",
    "answer_correct",
    "used_source_citations",
    "citation_correct",
    "abstained",
    "abstention_correct",
    "retrieval_time_seconds",
    "generation_time_seconds",
    "response_time_seconds",
]

CHUNK_DETAIL_FIELDNAMES = [
    "run_id",
    "configuration",
    "chunking_strategy",
    "top_k",
    "relevance_threshold",
    "question_id",
    "question",
    "is_answerable",
    "expected_source",
    "pinecone_rank",
    "pinecone_score",
    "sent_to_llm",
    "llm_source_number",
    "source",
    "document_name",
    "chunk_number",
    "chunk_word_count",
    "chunk_text",
    "is_expected_source",
    "is_irrelevant_context",
]


def load_evaluation_questions(csv_path: str | Path) -> list[EvaluationQuestion]:
    """Load evaluation questions from a CSV file."""
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_columns(reader.fieldnames or [])

        questions = []
        for row in reader:
            row["is_answerable"] = _to_bool(row.get("is_answerable", ""))
            questions.append(row)

    return questions


def check_retrieval_hit(
    retrieved_chunks: list[dict[str, Any]], expected_source: str
) -> bool:
    """Check whether retrieval returned the expected document source."""
    for chunk in retrieved_chunks:
        if _chunk_matches_expected_source(chunk, expected_source):
            return True

    return False


def check_answer_correctness(answer: str, expected_answer: str) -> bool:
    """Check whether the answer contains the expected key idea or phrases."""
    expected_answer = expected_answer.strip()
    if not expected_answer:
        return False

    normalized_answer = _normalize_for_matching(answer)
    expected_phrases = [
        _normalize_for_matching(phrase)
        for phrase in expected_answer.split(";")
        if phrase.strip()
    ]

    return all(phrase in normalized_answer for phrase in expected_phrases)


def check_citation_correctness(answer: str, sources: list[dict[str, Any]]) -> bool:
    """Check whether the answer cites at least one source that exists."""
    for source in sources:
        source_number = source.get("source_number")
        if source_number and f"[Source {source_number}]" in answer:
            return True

    return False


def check_abstention_correctness(
    answer: str, should_abstain: bool, abstained: bool | None = None
) -> bool:
    """Check whether the system abstained only when it should."""
    did_abstain = bool(abstained) or ABSTAIN_MESSAGE in answer
    return did_abstain == should_abstain


def evaluate_single_result(
    question: EvaluationQuestion,
    retrieval_response: dict[str, Any],
    generated_answer: dict[str, Any],
    response_time_seconds: float,
    configuration_name: str,
    run_id: str = "",
    chunking_strategy: str = "",
    namespace: str = "",
    top_k: int | str = "",
    relevance_threshold: float | str = "",
    retrieval_time_seconds: float | str = "",
    generation_time_seconds: float | str = "",
) -> EvaluationRecord:
    """Create one evaluation row for one question and one configuration."""
    answer = str(generated_answer.get("answer", ""))
    retrieved_chunks = retrieval_response.get("chunks", [])
    raw_chunks = retrieval_response.get("raw_chunks") or retrieved_chunks
    sources = generated_answer.get("sources", [])
    should_abstain = not bool(question["is_answerable"])
    expected_source = str(question.get("expected_source", ""))
    expected_source_rank, expected_source_chunk = find_expected_source(
        raw_chunks, expected_source
    )
    retrieval_hit = check_retrieval_hit(retrieved_chunks, expected_source)
    expected_source_score = (
        expected_source_chunk.get("score", "") if expected_source_chunk else ""
    )

    return {
        "run_id": run_id,
        "configuration": configuration_name,
        "chunking_strategy": chunking_strategy,
        "namespace": namespace or retrieval_response.get("namespace", ""),
        "top_k": top_k or retrieval_response.get("top_k", ""),
        "relevance_threshold": relevance_threshold
        if relevance_threshold != ""
        else retrieval_response.get("relevance_threshold", ""),
        "question_id": question.get("question_id"),
        "question": question.get("question"),
        "is_answerable": question.get("is_answerable"),
        "expected_answer": question.get("expected_answer"),
        "expected_source": expected_source,
        "raw_retrieved_count": retrieval_response.get(
            "raw_retrieved_count", len(raw_chunks)
        ),
        "chunks_sent_to_llm": retrieval_response.get(
            "filtered_count", len(retrieved_chunks)
        ),
        "retrieval_hit": retrieval_hit,
        "expected_source_rank": expected_source_rank,
        "expected_source_score": expected_source_score,
        "expected_source_filtered_out": bool(
            expected_source and expected_source_chunk and not retrieval_hit
        ),
        "generated_answer": answer,
        "answer_correct": check_answer_correctness(
            answer, str(question.get("expected_answer", ""))
        ),
        "used_source_citations": " | ".join(
            str(source.get("citation", "")) for source in sources
        ),
        "citation_correct": check_citation_correctness(answer, sources),
        "abstained": bool(generated_answer.get("abstained")),
        "abstention_correct": check_abstention_correctness(
            answer, should_abstain, generated_answer.get("abstained")
        ),
        "retrieval_time_seconds": retrieval_time_seconds,
        "generation_time_seconds": generation_time_seconds,
        "response_time_seconds": round(response_time_seconds, 3),
    }


def run_evaluation(
    questions_csv_path: str | Path = DEFAULT_QUESTIONS_PATH,
    results_output_path: str | Path = DEFAULT_RESULTS_PATH,
    chunk_details_output_path: str | Path = DEFAULT_CHUNK_DETAILS_PATH,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    retrieve_function: Any | None = None,
    generate_function: Any | None = None,
) -> dict[str, Any]:
    """Run every evaluation question through every planned configuration.

    The default version uses the real Pinecone retriever and OpenAI generator.
    Tests can pass fake functions so we can check the logic without API calls.
    """
    from rag.generator import build_abstention_response, generate_answer
    from rag.retriever import retrieve_relevant_chunks

    retrieve_function = retrieve_function or retrieve_relevant_chunks
    generate_function = generate_function or generate_answer

    questions = load_evaluation_questions(questions_csv_path)
    configurations = get_evaluation_configurations()
    run_id = create_run_id()

    result_records = []
    chunk_detail_records = []

    for configuration in configurations:
        for question in questions:
            result_record, chunk_records = _run_one_evaluation_case(
                question=question,
                configuration=configuration,
                relevance_threshold=relevance_threshold,
                max_output_tokens=max_output_tokens,
                retrieve_function=retrieve_function,
                generate_function=generate_function,
                abstain_function=build_abstention_response,
                run_id=run_id,
            )
            result_records.append(result_record)
            chunk_detail_records.extend(chunk_records)

    save_evaluation_results(
        result_records,
        results_output_path,
        fieldnames=RESULT_FIELDNAMES,
    )
    save_chunk_detail_records(chunk_detail_records, chunk_details_output_path)

    return {
        "run_id": run_id,
        "results": result_records,
        "chunk_details": chunk_detail_records,
        "summary": summarize_results(result_records),
        "results_path": str(results_output_path),
        "chunk_details_path": str(chunk_details_output_path),
    }


def _run_one_evaluation_case(
    question: EvaluationQuestion,
    configuration: dict[str, Any],
    relevance_threshold: float,
    max_output_tokens: int,
    retrieve_function: Any,
    generate_function: Any,
    abstain_function: Any,
    run_id: str,
) -> tuple[EvaluationRecord, list[EvaluationRecord]]:
    """Run one question with one configuration and return both CSV row types."""
    start_time = time.perf_counter()

    retrieval_start = time.perf_counter()
    retrieval_response = retrieve_function(
        question=str(question.get("question", "")),
        namespace=configuration["namespace"],
        top_k=configuration["top_k"],
        relevance_threshold=relevance_threshold,
    )
    retrieval_time = time.perf_counter() - retrieval_start

    generation_start = time.perf_counter()
    if retrieval_response.get("chunks"):
        generated_answer = generate_function(
            question=str(question.get("question", "")),
            retrieved_chunks=retrieval_response["chunks"],
            max_output_tokens=max_output_tokens,
        )
    else:
        generated_answer = abstain_function()
    generation_time = time.perf_counter() - generation_start

    total_time = time.perf_counter() - start_time
    result_record = evaluate_single_result(
        question=question,
        retrieval_response=retrieval_response,
        generated_answer=generated_answer,
        response_time_seconds=total_time,
        configuration_name=configuration["name"],
        run_id=run_id,
        chunking_strategy=configuration["chunking_strategy"],
        namespace=configuration["namespace"],
        top_k=configuration["top_k"],
        relevance_threshold=relevance_threshold,
        retrieval_time_seconds=round(retrieval_time, 3),
        generation_time_seconds=round(generation_time, 3),
    )

    chunk_records = build_chunk_detail_records(
        question=question,
        retrieval_response=retrieval_response,
        configuration_name=configuration["name"],
        run_id=run_id,
        chunking_strategy=configuration["chunking_strategy"],
        top_k=configuration["top_k"],
    )
    return result_record, chunk_records


def build_chunk_detail_records(
    question: EvaluationQuestion,
    retrieval_response: dict[str, Any],
    configuration_name: str,
    run_id: str = "",
    chunking_strategy: str = "",
    top_k: int | str = "",
) -> list[EvaluationRecord]:
    """Create CSV rows that show what happened to every retrieved chunk.

    Each row represents one chunk returned by Pinecone. The ``sent_to_llm``
    column tells us whether that chunk passed the threshold and became part of
    the prompt sent to the language model.
    """
    raw_chunks = retrieval_response.get("raw_chunks") or retrieval_response.get(
        "chunks", []
    )
    sent_chunks = retrieval_response.get("chunks", [])

    # Save the final LLM source number for chunks that passed the threshold.
    sent_positions = {
        _chunk_key(chunk): source_number
        for source_number, chunk in enumerate(sent_chunks, start=1)
    }

    records = []
    expected_source = str(question.get("expected_source", ""))
    for retrieved_rank, chunk in enumerate(raw_chunks, start=1):
        source_number = sent_positions.get(_chunk_key(chunk))
        is_expected_source = _chunk_matches_expected_source(chunk, expected_source)

        records.append(
            {
                "run_id": run_id,
                "configuration": configuration_name,
                "chunking_strategy": chunking_strategy
                or chunk.get("chunking_strategy", ""),
                "top_k": top_k or retrieval_response.get("top_k", ""),
                "relevance_threshold": retrieval_response.get("relevance_threshold"),
                "question_id": question.get("question_id"),
                "question": question.get("question"),
                "is_answerable": question.get("is_answerable"),
                "expected_source": expected_source,
                "pinecone_rank": retrieved_rank,
                "pinecone_score": chunk.get("score"),
                "sent_to_llm": bool(source_number),
                "llm_source_number": source_number or "",
                "source": chunk.get("citation"),
                "document_name": chunk.get("document_name"),
                "chunk_number": chunk.get("chunk_number"),
                "chunk_word_count": chunk.get("word_count")
                or len(str(chunk.get("text", "")).split()),
                "chunk_text": chunk.get("text", ""),
                "is_expected_source": is_expected_source,
                "is_irrelevant_context": bool(source_number) and not is_expected_source,
            }
        )

    return records


def summarize_results(records: list[EvaluationRecord]) -> dict[str, Any]:
    """Calculate simple average metrics for a list of evaluation records."""
    if not records:
        return {}

    metric_names = [
        "retrieval_hit",
        "answer_correct",
        "citation_correct",
        "abstention_correct",
    ]

    summary = {"total_questions": len(records)}
    for metric in metric_names:
        summary[f"{metric}_rate"] = _average([record[metric] for record in records])

    summary["average_response_time_seconds"] = _average(
        [record["response_time_seconds"] for record in records]
    )
    return summary


def save_evaluation_results(
    records: list[EvaluationRecord],
    output_path: str | Path,
    fieldnames: list[str] | None = None,
) -> None:
    """Save evaluation records to a CSV file."""
    if not records:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames or list(records[0].keys()),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)


def save_chunk_detail_records(
    records: list[EvaluationRecord], output_path: str | Path
) -> None:
    """Save the retrieved/sent chunk audit records to a CSV file."""
    save_evaluation_results(
        records,
        output_path,
        fieldnames=CHUNK_DETAIL_FIELDNAMES,
    )


def get_evaluation_configurations() -> list[dict[str, Any]]:
    """Return the four planned experiment settings."""
    return [
        {
            "name": "fixed_top_3",
            "chunking_strategy": "fixed_size",
            "namespace": "fixed_chunks",
            "top_k": 3,
        },
        {
            "name": "fixed_top_5",
            "chunking_strategy": "fixed_size",
            "namespace": "fixed_chunks",
            "top_k": 5,
        },
        {
            "name": "paragraph_top_3",
            "chunking_strategy": "paragraph_aware",
            "namespace": "paragraph_chunks",
            "top_k": 3,
        },
        {
            "name": "paragraph_top_5",
            "chunking_strategy": "paragraph_aware",
            "namespace": "paragraph_chunks",
            "top_k": 5,
        },
    ]


def create_run_id() -> str:
    """Create one stable identifier for every row in a single evaluation run."""
    return time.strftime("eval_%Y%m%d_%H%M%S")


def find_expected_source(
    chunks: list[dict[str, Any]], expected_source: str
) -> tuple[int | str, dict[str, Any] | None]:
    """Return the raw retrieval rank and chunk matching the expected source."""
    for rank, chunk in enumerate(chunks, start=1):
        if _chunk_matches_expected_source(chunk, expected_source):
            return rank, chunk

    return "", None


def _validate_columns(fieldnames: list[str]) -> None:
    """Make sure the evaluation CSV has the columns this project expects."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing_columns:
        raise ValueError(f"Missing evaluation CSV columns: {missing_columns}")


def _to_bool(value: str) -> bool:
    """Convert common CSV boolean text into a Python boolean."""
    return value.strip().lower() in ["true", "1", "yes", "y"]


def _chunk_matches_expected_source(
    chunk: dict[str, Any], expected_source: str
) -> bool:
    """Check whether a retrieved chunk points at the expected citation text."""
    expected_source = expected_source.strip().lower()
    if not expected_source:
        return False

    searchable_source_text = " ".join(
        str(chunk.get(key, ""))
        for key in ["citation", "document_name", "source_label", "id"]
    ).lower()

    return expected_source in searchable_source_text


def _chunk_key(chunk: dict[str, Any]) -> str:
    """Create a stable key so raw retrieved chunks can be matched to LLM chunks."""
    return str(
        chunk.get("id")
        or chunk.get("chunk_id")
        or f"{chunk.get('citation', '')}|{chunk.get('text', '')}"
    )


def _average(values: list[Any]) -> float:
    """Return the average of booleans or numbers as a rounded float."""
    return round(sum(float(value) for value in values) / len(values), 3)


def _normalize_for_matching(text: str) -> str:
    """Make expected-answer checks insensitive to punctuation and extra spacing."""
    normalized_characters = [
        character.lower() if character.isalnum() else " " for character in str(text)
    ]
    return " ".join("".join(normalized_characters).split())


def main() -> None:
    """Run the evaluation from the terminal with the default settings."""
    print("Running evaluation. This may call Pinecone and OpenAI.")
    report = run_evaluation()
    print(f"Saved results to: {report['results_path']}")
    print(f"Saved chunk details to: {report['chunk_details_path']}")
    print(f"Summary: {report['summary']}")


if __name__ == "__main__":
    main()
