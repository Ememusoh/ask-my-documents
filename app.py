"""Streamlit app for the Ask My Documents prototype."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from rag.chunking import fixed_size_chunks, paragraph_aware_chunks
from rag.embeddings import delete_document_chunks, index_chunks
from rag.extractors import extract_document
from rag.generator import build_abstention_response, generate_answer
from rag.retriever import filter_prepared_chunks, retrieve_relevant_chunks


UPLOAD_DIR = Path("data/uploads")
INDEX_LOG_PATH = Path("data/index/indexed_documents.json")
CHUNKING_VERSION = "document_level_fixed_size_v1"


def main() -> None:
    """Run the Streamlit application."""
    load_dotenv()
    setup_page()
    setup_session_state()

    settings = show_sidebar_settings()
    st.title("Ask My Documents")
    st.write("Upload a document, index its chunks, then ask grounded questions.")

    uploaded_file = st.file_uploader("Upload a TXT, PDF, or DOCX file", type=["txt", "pdf", "docx"])

    if st.button("Process document", disabled=uploaded_file is None):
        process_uploaded_document(uploaded_file, settings)

    show_processing_status()

    if st.session_state.chunks:
        st.checkbox("Force re-index even if this document was indexed before", key="force_reindex")
        if st.button("Index chunks in Pinecone"):
            index_current_chunks()

        ask_question(settings)


def setup_page() -> None:
    """Set simple page settings for the prototype."""
    st.set_page_config(page_title="Ask My Documents", page_icon="📄", layout="wide")


def setup_session_state() -> None:
    """Store processed data so Streamlit reruns do not erase progress."""
    defaults = {
        "document_path": None,
        "sections": [],
        "chunks": [],
        "chunking_strategy": None,
        "namespace": None,
        "index_fingerprint": None,
        "indexed": False,
        "last_answer": None,
        "last_retrieval": None,
        "last_retrieval_signature": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_sidebar_settings() -> dict[str, Any]:
    """Collect the user-controlled RAG settings."""
    st.sidebar.header("RAG settings")

    chunking_strategy = st.sidebar.selectbox(
        "Chunking strategy",
        ["fixed_size", "paragraph_aware"],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    chunk_size = st.sidebar.number_input("Chunk size words", min_value=20, value=120, step=10)

    overlap = 0
    if chunking_strategy == "fixed_size":
        overlap = st.sidebar.number_input("Overlap words", min_value=0, value=20, step=5)

    top_k = st.sidebar.number_input("Number of chunks to retrieve", min_value=1, value=3, step=1)
    relevance_threshold = st.sidebar.slider(
        "Relevance threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )
    max_output_tokens = st.sidebar.number_input(
        "Max answer tokens", min_value=100, value=300, step=50
    )

    return {
        "chunking_strategy": chunking_strategy,
        "chunk_size": int(chunk_size),
        "overlap": int(overlap),
        "top_k": int(top_k),
        "relevance_threshold": float(relevance_threshold),
        "max_output_tokens": int(max_output_tokens),
    }


def process_uploaded_document(uploaded_file: Any, settings: dict[str, Any]) -> None:
    """Save, extract, and chunk the uploaded document."""
    try:
        file_path = save_uploaded_file(uploaded_file)
        sections = extract_document(file_path)
        chunks = create_chunks(sections, settings)
        namespace = get_namespace(settings["chunking_strategy"])
        fingerprint = create_index_fingerprint(file_path, settings)

        st.session_state.document_path = file_path
        st.session_state.sections = sections
        st.session_state.chunks = chunks
        st.session_state.chunking_strategy = settings["chunking_strategy"]
        st.session_state.namespace = namespace
        st.session_state.index_fingerprint = fingerprint
        st.session_state.indexed = is_already_indexed(fingerprint)
        st.session_state.last_answer = None
        st.session_state.last_retrieval = None
        st.session_state.last_retrieval_signature = None

        st.success(f"Processed {file_path.name}: {len(sections)} sections, {len(chunks)} chunks.")
        if st.session_state.indexed:
            st.info("This document and chunking setup was already indexed in Pinecone.")
    except Exception as error:
        st.error(f"Could not process document: {error}")


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Save the uploaded file locally so the extractors can read it."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


def create_chunks(sections: list[dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Create chunks using the selected chunking strategy."""
    if settings["chunking_strategy"] == "fixed_size":
        return fixed_size_chunks(
            sections,
            chunk_size=settings["chunk_size"],
            overlap=settings["overlap"],
        )

    return paragraph_aware_chunks(sections, chunk_size=settings["chunk_size"])


def get_namespace(chunking_strategy: str) -> str:
    """Choose the Pinecone namespace for the selected chunking strategy."""
    if chunking_strategy == "fixed_size":
        return os.getenv("PINECONE_NAMESPACE_FIXED", "fixed_chunks")

    return os.getenv("PINECONE_NAMESPACE_PARAGRAPH", "paragraph_chunks")


def show_processing_status() -> None:
    """Show the current document, chunk, and indexing status."""
    if not st.session_state.chunks:
        return

    st.subheader("Processed document")
    st.write(f"Document: `{Path(st.session_state.document_path).name}`")
    st.write(f"Chunking strategy: `{st.session_state.chunking_strategy}`")
    st.write(f"Pinecone namespace: `{st.session_state.namespace}`")
    st.write(f"Chunks created: `{len(st.session_state.chunks)}`")
    st.write(f"Indexed in Pinecone: `{st.session_state.indexed}`")

    with st.expander("Preview first chunks"):
        for chunk in st.session_state.chunks[:3]:
            st.markdown(f"**Chunk {chunk['chunk_number']}** — {chunk['word_count']} words")
            st.write(chunk["text"][:700])


def index_current_chunks() -> None:
    """Upload the current chunks to Pinecone."""
    try:
        force_reindex = st.session_state.get("force_reindex", False)
        if st.session_state.indexed and not force_reindex:
            st.info("Already indexed. You can ask questions now.")
            return

        with st.spinner("Indexing chunks in Pinecone..."):
            delete_document_chunks(
                document_name=Path(st.session_state.document_path).name,
                namespace=st.session_state.namespace,
            )
            indexed_count = index_chunks(
                st.session_state.chunks,
                namespace=st.session_state.namespace,
            )

        mark_as_indexed(st.session_state.index_fingerprint)
        st.session_state.indexed = True
        st.success(f"Indexed {indexed_count} chunks into `{st.session_state.namespace}`.")
    except Exception as error:
        st.error(f"Could not index chunks: {error}")


def ask_question(settings: dict[str, Any]) -> None:
    """Retrieve evidence first, then generate an answer from filtered evidence."""
    st.subheader("Ask a question")
    question = st.text_input("Question")

    retrieve_column, generate_column = st.columns(2)
    with retrieve_column:
        retrieve_clicked = st.button("Retrieve sources from Pinecone")
    with generate_column:
        generate_clicked = st.button("Generate answer")

    if retrieve_clicked:
        retrieve_sources_for_question(question, settings)

    display_raw_retrieval(settings)

    if generate_clicked:
        generate_answer_from_retrieval(question, settings)

    display_last_answer()


def retrieve_sources_for_question(question: str, settings: dict[str, Any]) -> None:
    """Retrieve top-k chunks from Pinecone and store both raw and filtered results."""
    try:
        if not st.session_state.indexed:
            st.warning("Index the chunks in Pinecone before asking a question.")
            return
        if not question.strip():
            st.warning("Enter a question first.")
            return

        start_time = time.perf_counter()
        retrieval = retrieve_relevant_chunks(
            question=question,
            namespace=st.session_state.namespace,
            top_k=settings["top_k"],
            relevance_threshold=0.0,
        )
        retrieval["retrieval_time_seconds"] = round(time.perf_counter() - start_time, 3)

        st.session_state.last_retrieval = retrieval
        st.session_state.last_retrieval_signature = make_retrieval_signature(
            question, settings
        )
        st.session_state.last_answer = None
        st.success("Retrieved sources from Pinecone.")
    except Exception as error:
        st.error(f"Could not retrieve sources: {error}")


def generate_answer_from_retrieval(question: str, settings: dict[str, Any]) -> None:
    """Generate an answer using only chunks that passed the threshold."""
    retrieval = st.session_state.last_retrieval
    if not retrieval:
        st.warning("Click 'Retrieve sources from Pinecone' before generating an answer.")
        return

    current_signature = make_retrieval_signature(question, settings)
    if current_signature != st.session_state.last_retrieval_signature:
        st.warning("Question, top-k, or namespace changed. Retrieve sources again first.")
        return

    try:
        start_time = time.perf_counter()
        filtered_chunks = chunks_for_current_threshold(retrieval, settings)
        if filtered_chunks:
            answer = generate_answer(
                question=question,
                retrieved_chunks=filtered_chunks,
                max_output_tokens=settings["max_output_tokens"],
            )
        else:
            answer = build_abstention_response()

        update_retrieval_after_threshold(retrieval, filtered_chunks, settings)
        answer["response_time_seconds"] = round(time.perf_counter() - start_time, 3)
        st.session_state.last_answer = answer
    except Exception as error:
        st.error(f"Could not generate answer: {error}")


def make_retrieval_signature(question: str, settings: dict[str, Any]) -> tuple[Any, ...]:
    """Remember which question and settings produced the current retrieval."""
    return (
        question.strip(),
        st.session_state.namespace,
        settings["top_k"],
    )


def display_raw_retrieval(settings: dict[str, Any]) -> None:
    """Show Pinecone results before threshold filtering."""
    retrieval = st.session_state.last_retrieval
    if not retrieval:
        return

    filtered_chunks = chunks_for_current_threshold(retrieval, settings)
    update_retrieval_after_threshold(retrieval, filtered_chunks, settings)

    st.subheader("Sources retrieved from Pinecone before threshold filtering")
    st.write(
        f"Pinecone returned `{retrieval['raw_retrieved_count']}` chunks in "
        f"`{retrieval['retrieval_time_seconds']}` seconds."
    )
    st.write(
        f"Current threshold: `{settings['relevance_threshold']}`. "
        f"Chunks that would be sent to the LLM: `{len(filtered_chunks)}`."
    )
    display_retrieved_chunks(retrieval["raw_chunks"])


def chunks_for_current_threshold(
    retrieval: dict[str, Any], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply the current threshold to stored Pinecone results locally."""
    return filter_prepared_chunks(
        retrieval["raw_chunks"],
        relevance_threshold=settings["relevance_threshold"],
    )


def update_retrieval_after_threshold(
    retrieval: dict[str, Any],
    filtered_chunks: list[dict[str, Any]],
    settings: dict[str, Any],
) -> None:
    """Store which chunks pass the current threshold without re-querying Pinecone."""
    retrieval["chunks"] = filtered_chunks
    retrieval["filtered_count"] = len(filtered_chunks)
    retrieval["retrieved_count"] = len(filtered_chunks)
    retrieval["can_answer"] = bool(filtered_chunks)
    retrieval["relevance_threshold"] = settings["relevance_threshold"]


def display_last_answer() -> None:
    """Display the latest answer and its supporting sources."""
    answer = st.session_state.last_answer
    if not answer:
        return

    st.subheader("Answer")
    st.write(answer["answer"])
    st.caption(f"Response time: {answer['response_time_seconds']} seconds")

    if answer["sources"]:
        st.subheader("Sources sent to the LLM after threshold filtering")
        display_sources(answer["sources"])
    elif st.session_state.last_retrieval:
        st.info("No chunks passed the threshold, so no sources were sent to the LLM.")


def display_sources(sources: list[dict[str, Any]]) -> None:
    """Show source metadata and passages outside the OpenAI prompt."""
    for source in sources:
        with st.expander(f"{source['label']} — {source['citation']}"):
            st.write(f"Similarity score: `{source['score']}`")
            st.write(source["text"])


def display_retrieved_chunks(chunks: list[dict[str, Any]]) -> None:
    """Show retrieved chunks with scores and citation labels."""
    if not chunks:
        st.write("No chunks to display.")
        return

    for index, chunk in enumerate(chunks, start=1):
        score = round(float(chunk.get("score", 0.0)), 3)
        citation = chunk.get("citation", "Unknown source")
        with st.expander(f"Chunk {index} — score {score} — {citation}"):
            st.write(chunk.get("text", ""))


def create_index_fingerprint(file_path: Path, settings: dict[str, Any]) -> str:
    """Create a stable ID for one document plus one chunking setup."""
    file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    fingerprint_text = "|".join(
        [
            file_hash,
            settings["chunking_strategy"],
            str(settings["chunk_size"]),
            str(settings["overlap"]),
            get_namespace(settings["chunking_strategy"]),
            CHUNKING_VERSION,
        ]
    )
    return hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()


def is_already_indexed(fingerprint: str) -> bool:
    """Check the local index log to avoid re-indexing the same setup."""
    indexed_documents = load_index_log()
    return fingerprint in indexed_documents


def mark_as_indexed(fingerprint: str) -> None:
    """Remember that this document setup has already been sent to Pinecone."""
    indexed_documents = load_index_log()
    indexed_documents[fingerprint] = {"indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    INDEX_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_LOG_PATH.write_text(json.dumps(indexed_documents, indent=2), encoding="utf-8")


def load_index_log() -> dict[str, Any]:
    """Load the local index log, returning an empty log if it does not exist yet."""
    if not INDEX_LOG_PATH.exists():
        return {}

    try:
        return json.loads(INDEX_LOG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


if __name__ == "__main__":
    main()
