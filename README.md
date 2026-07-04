# Ask My Documents

Ask My Documents is an academic Retrieval-Augmented Generation (RAG) prototype. Users will
upload PDF, DOCX, and TXT files, ask questions about them, and receive answers grounded only in
retrieved document passages. Each answer will display supporting source information so it can be
checked.

The project uses one configurable RAG pipeline. Its chunking strategy, retrieval depth, and
relevance threshold will be varied for evaluation rather than implemented as separate systems.

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Add API credentials only to `.env`; it is ignored by Git.

## Run the evaluation

Make sure your documents have already been indexed in Pinecone for both chunking namespaces.
Then run:

```bash
PYTHONPATH=. uv run python -m rag.evaluator
```

This creates two CSV files:

- `data/results/evaluation_results.csv`: one row per question and configuration
- `data/results/evaluation_chunk_details.csv`: one row per retrieved chunk, including whether it was sent to the LLM

Running the evaluation can call Pinecone and OpenAI, so use a small question set while testing.

## Planned development milestones

1. Text extraction for TXT, PDF, and DOCX
2. Fixed-size and paragraph-aware chunking
3. Embedding generation
4. Cosine-similarity retrieval
5. Grounded answer generation with citations
6. Streamlit interface
7. Evaluation script for four configurations:
   - fixed chunking + top 3
   - fixed chunking + top 5
   - paragraph-aware chunking + top 3
   - paragraph-aware chunking + top 5

## Project layout

```text
app.py                  Streamlit entry point
rag/                    RAG pipeline modules
data/uploads/           Local uploaded documents (ignored by Git)
data/index/             Generated embeddings and metadata (ignored by Git)
data/evaluation/        Evaluation question set
data/results/           Generated evaluation outputs (ignored by Git)
tests/                  Unit-test scaffolds
```
