"""Tests for grounded-profile chunking and upsert rules."""

from unittest.mock import MagicMock

from app.pipelines.profile_chunks import (
    SOURCE_ACHIEVEMENT,
    SOURCE_EXPERIENCE,
    SOURCE_PROJECT,
    facts_to_chunks,
    retrieve_chunks,
    upsert_profile_chunks,
)


PROFILE = {
    "skills": ["Python", "PostgreSQL", "RAG"],
    "experience": [
        {
            "title": "Backend Engineer",
            "company": "Acme",
            "start_date": "2022",
            "bullets": [
                "Shipped pgvector search over 12k embeddings in PostgreSQL.",
                "Cut p95 latency 40% on the match API.",
            ],
        }
    ],
    "projects": ["Built a RAG eval harness for cover-letter drafts."],
    "education": ["BSc Computer Science"],
    "certifications": [],
    "volunteering": [],
}


def test_facts_to_chunks_splits_roles_bullets_and_projects():
    chunks = facts_to_chunks(PROFILE)
    types = {chunk["source_type"] for chunk in chunks}
    assert SOURCE_EXPERIENCE in types
    assert SOURCE_ACHIEVEMENT in types
    assert SOURCE_PROJECT in types
    assert any("pgvector" in chunk["chunk_text"] for chunk in chunks)
    ids = [chunk["source_id"] for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_facts_to_chunks_is_stable_across_calls():
    first = {chunk["source_id"]: chunk["content_hash"] for chunk in facts_to_chunks(PROFILE)}
    second = {chunk["source_id"]: chunk["content_hash"] for chunk in facts_to_chunks(PROFILE)}
    assert first == second


def test_upsert_skips_reembed_when_hash_matches(monkeypatch):
    existing = facts_to_chunks(PROFILE)
    for index, chunk in enumerate(existing, start=1):
        chunk["id"] = index
        chunk["user_id"] = 7
        chunk["embedding"] = [0.1] * 1536
        chunk["is_active"] = True

    xano = MagicMock()
    xano.list_chunks.return_value = existing
    monkeypatch.setattr("app.pipelines.profile_chunks.xano", xano)
    monkeypatch.setattr(
        "app.pipelines.profile_chunks.embed_texts",
        lambda texts, task_type=None: [_ for _ in texts],
    )

    stats = upsert_profile_chunks(7, PROFILE)
    assert stats["inserted"] == 0
    assert stats["updated"] == 0
    assert stats["embedded"] == 0
    xano.add_chunk.assert_not_called()
    xano.patch_chunk.assert_not_called()


def test_upsert_deactivates_removed_facts(monkeypatch):
    stale = {
        "id": 99,
        "user_id": 7,
        "source_type": SOURCE_PROJECT,
        "source_id": "proj:gone",
        "chunk_text": "Old project",
        "content_hash": "abc",
        "is_active": True,
        "embedding": [0.1] * 1536,
    }
    xano = MagicMock()
    xano.list_chunks.return_value = [stale]
    monkeypatch.setattr("app.pipelines.profile_chunks.xano", xano)
    monkeypatch.setattr(
        "app.pipelines.profile_chunks.embed_texts",
        lambda texts, task_type=None: [[0.2] * 1536 for _ in texts],
    )

    stats = upsert_profile_chunks(7, PROFILE)
    assert stats["deactivated"] == 1
    xano.patch_chunk.assert_any_call(99, {"is_active": False})


def test_retrieve_falls_back_to_local_chunks(monkeypatch):
    monkeypatch.setattr("app.pipelines.profile_chunks.embed_text", lambda *args, **kwargs: [])
    chunks = facts_to_chunks(PROFILE)
    rows, source = retrieve_chunks(
        None,
        "Need pgvector and RAG on PostgreSQL",
        local_chunks=chunks,
    )
    assert source == "local"
    assert rows
    assert any("pgvector" in row["chunk_text"].lower() or "rag" in row["chunk_text"].lower() for row in rows)
