# Ask My Documents

Ask My Documents is a local Retrieval-Augmented Generation (RAG) prototype for
question answering over uploaded documents. A user can upload PDF, DOCX, or TXT
files, ask questions about them, and receive answers based only on retrieved
document passages.

The project focuses on transparency. Instead of returning only a generated
answer, the app shows the retrieved chunks, similarity scores, source metadata,
threshold filtering decisions, and citations used to support the response.

## Project Goal

The goal of this project is to design, implement, and evaluate a small
configurable RAG system for uploaded documents.

The prototype demonstrates:

- document upload and text extraction for PDF, DOCX, and TXT files
- fixed-size and paragraph-aware chunking
- embedding and vector search using Pinecone
- top-k retrieval and relevance-threshold filtering
- grounded answer generation using OpenAI
- citation display and retrieved-source inspection
- evaluation across chunking and retrieval-depth configurations

## How The System Works

```text
Upload document
-> Extract text and source metadata
-> Split text into chunks
-> Index chunks in Pinecone
-> Ask a question
-> Retrieve top-k relevant chunks
-> Filter chunks by relevance threshold
-> Generate a grounded answer
-> Display answer with sources
```

Pinecone is used for vector indexing and search. The project-specific work is
the transparent RAG workflow around it: chunking, metadata handling, retrieval
settings, threshold filtering, prompt construction, citation display, and
evaluation.

## Setup

Create and activate a virtual environment:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Then add your own OpenAI and Pinecone credentials to `.env`.

Do not commit `.env`; it is ignored by Git.

## Run The App

From the project root:

```bash
.venv/bin/python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Suggested Demo Settings

For a simple demonstration:

```text
Chunking strategy: Fixed Size
Chunk size words: 120
Overlap words: 20
Number of chunks to retrieve: 3
Relevance threshold: 0.30
Max answer tokens: 600
```

For paragraph-aware chunking, the overlap setting is not used because chunks are
formed around paragraph boundaries.

## Evaluation

The evaluation question set is stored in:

```text
data/evaluation/eval_questions.csv
```

The evaluator compares four configurations:

- fixed-size chunking with top-3 retrieval
- fixed-size chunking with top-5 retrieval
- paragraph-aware chunking with top-3 retrieval
- paragraph-aware chunking with top-5 retrieval

To run the evaluation, make sure the documents have been indexed in the required
Pinecone namespaces, then run:

```bash
PYTHONPATH=. .venv/bin/python -m rag.evaluator
```

Evaluation outputs are written to `data/results/`, which is ignored by Git
because the files are generated artifacts.

## Final Report And Presentation

The final submission documents are included in:

```text
reports/Ask_My_Documents_Final_Report.pdf
reports/Ask_My_Documents_Final_Presentation.pptx
```

Draft reports, temporary files, local notebooks, tests, uploaded documents,
generated indexes, result CSVs, and credentials are intentionally excluded from
the GitHub submission.

## Repository Layout

```text
app.py                         Streamlit app entry point
rag/                           RAG pipeline modules
data/evaluation/               Evaluation question set
data/uploads/.gitkeep          Placeholder for local uploaded files
data/index/.gitkeep            Placeholder for local index metadata
data/results/.gitkeep          Placeholder for generated evaluation outputs
reports/                       Final report and presentation
requirements.txt               Python dependencies
.env.example                   Environment variable template
```

## Notes On Scope

This is an academic prototype, not a production document-management platform.
It does not include user authentication, deployment hardening, advanced OCR,
scanned-document understanding, complex table or diagram analysis, or large-scale
statistical validation.
