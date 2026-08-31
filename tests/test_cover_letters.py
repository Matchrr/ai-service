"""Tests for RAG cover-letter composition and persist payload."""

from unittest.mock import MagicMock

from app.pipelines import cover_letters as cover_module
from app.pipelines.cover_letters import compose_cover_letter, generate_cover_letter, verify_grounding
from app.pipelines.profile_chunks import facts_to_chunks


PROFILE = {
    "full_name": "Alex Rivera",
    "skills": ["Python", "PostgreSQL", "RAG"],
    "experience": [
        {
            "title": "Backend Engineer",
            "company": "Acme",
            "bullets": ["Shipped pgvector search over 12k embeddings in PostgreSQL."],
        }
    ],
    "projects": ["Built a RAG eval harness."],
    "education": [],
    "certifications": [],
    "volunteering": [],
}


def test_compose_letter_uses_retrieved_chunk_text():
    chunks = facts_to_chunks(PROFILE)
    letter, highlighted = compose_cover_letter(
        PROFILE,
        "Machine Learning Engineer",
        "Northstar",
        chunks,
        ["RAG", "PostgreSQL"],
        [],
    )
    assert "Northstar" in letter
    assert "pgvector" in letter.lower() or "rag" in letter.lower()
    assert "Alex Rivera" in letter
    assert "RAG" in highlighted or "PostgreSQL" in highlighted


def test_verify_grounding_rejects_unverified_skill():
    result = verify_grounding(PROFILE, ["I have shipped Kubernetes in production."])
    assert result["passed"] is False
    assert "Kubernetes" in result["rejected_claims"]


def test_generate_persists_letter_for_xano_user(monkeypatch):
    chunks = facts_to_chunks(PROFILE)
    for index, chunk in enumerate(chunks, start=1):
        chunk["id"] = index
        chunk["is_active"] = True
        chunk["embedding"] = [0.1] * 1536

    xano = MagicMock()
    xano.list_chunks.return_value = chunks
    xano.vector_search_chunks.return_value = None
    xano.list_cover_letters.return_value = []
    xano.add_cover_letter.return_value = {"id": 501}

    monkeypatch.setattr(cover_module, "xano", xano)
    monkeypatch.setattr(cover_module, "upsert_profile_chunks", lambda *args, **kwargs: {"inserted": 0})
    monkeypatch.setattr(cover_module, "retrieve_chunks", lambda *args, **kwargs: (chunks, "xano_list"))
    monkeypatch.setattr(cover_module, "embed_text", lambda *args, **kwargs: [0.1] * 1536)

    result = generate_cover_letter(
        {
            "user_id": 7,
            "xano_job_id": 42,
            "job_title": "Backend Engineer",
            "company": "Northstar",
            "job_description": "Need RAG and pgvector.",
            "matching_skills": ["RAG"],
            "missing_tech": [],
            "profile": PROFILE,
            "profile_revision_id": 1,
        }
    )
    assert result["persisted"] is True
    assert result["cover_letter_id"] == 501
    assert result["generation_source"] in {"rag", "fallback_template"}
    payload = xano.add_cover_letter.call_args[0][0]
    assert payload["user_id"] == 7
    assert payload["matchrr_job_position_id"] == 42
    assert payload["tone"] == "professional"
    assert payload["grounding_passed"] is True
    assert "retrieved_chunk_ids" in payload or "retreived_chunk_ids" in payload
