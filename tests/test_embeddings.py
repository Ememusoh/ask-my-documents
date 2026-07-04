"""Tests for Pinecone indexing helper behavior."""

from rag import embeddings


def test_index_chunks_uses_keyword_arguments(monkeypatch) -> None:
    """Pinecone upsert_records expects records and namespace as keyword arguments."""

    class FakeIndex:
        def __init__(self) -> None:
            self.calls = []

        def upsert_records(self, *, records, namespace):
            self.calls.append({"records": records, "namespace": namespace})

    fake_index = FakeIndex()
    monkeypatch.setattr(embeddings, "get_pinecone_index", lambda: fake_index)

    chunks = [
        {
            "chunk_id": "demo:chunk:1",
            "text": "This is a chunk.",
            "document_name": "demo.txt",
        }
    ]

    indexed_count = embeddings.index_chunks(chunks, namespace="fixed_chunks")

    assert indexed_count == 1
    assert fake_index.calls[0]["namespace"] == "fixed_chunks"


def test_search_response_reads_current_pinecone_score_names() -> None:
    """Current Pinecone responses use id_ and score_ in to_dict output."""

    class FakeResponse:
        def to_dict(self):
            return {
                "result": {
                    "hits": [
                        {
                            "id_": "chunk-1",
                            "score_": 0.38,
                            "fields": {"chunk_text": "Relevant passage."},
                        }
                    ]
                }
            }

    hits = embeddings._search_response_to_hits(FakeResponse())

    assert hits[0]["id"] == "chunk-1"
    assert hits[0]["score"] == 0.38
    assert hits[0]["chunk_text"] == "Relevant passage."


def test_delete_document_chunks_uses_document_filter(monkeypatch) -> None:
    """Re-indexing should delete old chunks for only the current document."""

    class FakeIndex:
        def __init__(self) -> None:
            self.delete_call = None

        def delete(self, *, namespace, filter):
            self.delete_call = {"namespace": namespace, "filter": filter}

    fake_index = FakeIndex()
    monkeypatch.setattr(embeddings, "get_pinecone_index", lambda: fake_index)

    embeddings.delete_document_chunks("demo.txt", namespace="fixed_chunks")

    assert fake_index.delete_call == {
        "namespace": "fixed_chunks",
        "filter": {"document_name": {"$eq": "demo.txt"}},
    }


def test_delete_document_chunks_ignores_missing_namespace(monkeypatch) -> None:
    """A new namespace may not exist yet, so that delete error should be harmless."""

    class FakeIndex:
        def delete(self, *, namespace, filter):
            raise Exception("[404] Namespace not found")

    monkeypatch.setattr(embeddings, "get_pinecone_index", lambda: FakeIndex())

    embeddings.delete_document_chunks("demo.txt", namespace="paragraph_chunks")
