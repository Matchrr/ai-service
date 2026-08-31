"""Split a grounded profile into `matchrr_profile_chunk` rows and embed them.

One row is one fact (a bullet, project, role, school, or cert). Cover-letter
generation retrieves these — never generated letters — filtered to this user.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

from app.core.config import settings
from app.db.xano import XanoError, xano
from app.lib.skills import extract_skills
from app.pipelines.embeddings import embed_text, embed_texts
from app.pipelines.jobs import content_hash

logger = logging.getLogger(__name__)

SOURCE_EXPERIENCE = "experience"
SOURCE_PROJECT = "project"
SOURCE_ACHIEVEMENT = "achievement"
SOURCE_EDUCATION = "education"
SOURCE_CERTIFICATION = "certification"

RETRIEVE_K = 8
_MAX_CHUNKS = 80


def _as_vector(value: Any) -> list[float]:
    if isinstance(value, list) and value and all(isinstance(item, (int, float)) for item in value[:8]):
        return [float(item) for item in value]
    return []


def _dense_cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for a, b in zip(left, right):
        dot += a * b
        norm_left += a * a
        norm_right += b * b
    if norm_left <= 0 or norm_right <= 0:
        return 0.0
    return dot / math.sqrt(norm_left * norm_right)


def _is_active(row: dict[str, Any]) -> bool:
    value = row.get("is_active")
    if value is None:
        return True
    return value not in {False, 0, "0", "false", "False"}


def _valid_embedding(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    vector = _as_vector(row.get("embedding"))
    expected = settings.embedding_dimensions or 1536
    return len(vector) == expected


def _stable_id(*parts: str) -> str:
    raw = "|".join(part.strip().lower() for part in parts if part and part.strip())
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _skill_list(text: str) -> list[str]:
    return extract_skills(text)


def facts_to_chunks(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic fact split. source_id is stable across re-grounds of the same text."""
    chunks: list[dict[str, Any]] = []

    def add(source_type: str, source_id: str, text: str, company: str = "") -> None:
        chunk_text = _clean(text)
        if not chunk_text:
            return
        chunks.append(
            {
                "source_type": source_type,
                "source_id": source_id,
                "chunk_text": chunk_text,
                "content_hash": content_hash(chunk_text),
                "skills": _skill_list(chunk_text),
                "company": _clean(company),
                "is_active": True,
            }
        )

    for role in list(profile.get("experience") or []) + list(profile.get("volunteering") or []):
        if not isinstance(role, dict):
            continue
        title = _clean(role.get("title"))
        company = _clean(role.get("company"))
        start = _clean(role.get("start_date"))
        if title or company:
            if title and company:
                role_text = f"{title} at {company}"
            else:
                role_text = title or company
            add(
                SOURCE_EXPERIENCE,
                f"exp:{_stable_id(company, title, start)}",
                role_text,
                company,
            )
        bullets = role.get("bullets") or []
        if not isinstance(bullets, list):
            continue
        for bullet in bullets:
            text = _clean(bullet)
            if not text:
                continue
            add(
                SOURCE_ACHIEVEMENT,
                f"ach:{_stable_id(company, title, text)}",
                text,
                company,
            )

    for project in profile.get("projects") or []:
        text = _clean(project)
        if text:
            add(SOURCE_PROJECT, f"proj:{_stable_id(text)}", text)

    for item in profile.get("education") or []:
        text = _clean(item)
        if text:
            add(SOURCE_EDUCATION, f"edu:{_stable_id(text)}", text)

    for item in profile.get("certifications") or []:
        text = _clean(item)
        if text:
            add(SOURCE_CERTIFICATION, f"cert:{_stable_id(text)}", text)

    # Deduplicate by source_id; first write wins.
    unique: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        unique.setdefault(chunk["source_id"], chunk)
    return list(unique.values())[:_MAX_CHUNKS]


def _index_existing(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row.get('source_type')}|{row.get('source_id')}"
        indexed[key] = row
    return indexed


def _chunk_payload(user_id: int, chunk: dict[str, Any], embedding: list[float] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_id": user_id,
        "source_type": chunk["source_type"],
        "source_id": chunk["source_id"],
        "chunk_text": chunk["chunk_text"],
        "content_hash": chunk["content_hash"],
        "skills": chunk.get("skills") or [],
        "is_active": True,
    }
    company = chunk.get("company") or ""
    if company:
        payload["company"] = company
    expected = settings.embedding_dimensions or 1536
    if embedding and len(embedding) == expected:
        payload["embedding"] = embedding
    return payload


def upsert_profile_chunks(user_id: int, profile: dict[str, Any]) -> dict[str, int]:
    """Write active chunks for this user. Stale facts are deactivated, not deleted."""
    desired = facts_to_chunks(profile)
    existing_rows = xano.list_chunks(user_id)
    existing = _index_existing(existing_rows)

    to_embed: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
    inserted = updated = embedded = deactivated = 0

    desired_keys: set[str] = set()
    for chunk in desired:
        key = f"{chunk['source_type']}|{chunk['source_id']}"
        desired_keys.add(key)
        row = existing.get(key)
        same_hash = bool(row) and str(row.get("content_hash") or "") == chunk["content_hash"]
        if same_hash and _valid_embedding(row) and _is_active(row):
            continue
        to_embed.append((key, chunk, row))

    texts = [item[1]["chunk_text"] for item in to_embed]
    vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT") if texts else []

    for (key, chunk, row), vector in zip(to_embed, vectors):
        payload = _chunk_payload(user_id, chunk, vector or None)
        try:
            if row and row.get("id") is not None:
                xano.patch_chunk(row["id"], payload)
                updated += 1
            else:
                xano.add_chunk(payload)
                inserted += 1
        except XanoError:
            logger.exception("Could not upsert profile chunk %s", key)
            continue
        if vector:
            embedded += 1

    for row in existing_rows:
        key = f"{row.get('source_type')}|{row.get('source_id')}"
        if key in desired_keys:
            continue
        if not _is_active(row):
            continue
        record_id = row.get("id")
        if record_id is None:
            continue
        try:
            xano.patch_chunk(record_id, {"is_active": False})
            deactivated += 1
        except XanoError:
            logger.exception("Could not deactivate profile chunk %s", record_id)

    return {
        "user_id": user_id,
        "desired": len(desired),
        "inserted": inserted,
        "updated": updated,
        "embedded": embedded,
        "deactivated": deactivated,
    }


def _lexical_score(chunk: dict[str, Any], job_text: str, job_skills: list[str]) -> float:
    owned = {skill.lower() for skill in (chunk.get("skills") or [])}
    overlap = sum(1 for skill in job_skills if skill.lower() in owned)
    text = (chunk.get("chunk_text") or "").lower()
    job_tokens = set(job_text.lower().split())
    hit = sum(1 for token in text.split() if len(token) > 3 and token in job_tokens)
    return overlap * 2.0 + min(hit, 6) * 0.15


def retrieve_chunks(
    user_id: int | None,
    job_text: str,
    k: int = RETRIEVE_K,
    local_chunks: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return top-k evidence chunks and the retrieval source label."""
    job_skills = extract_skills(job_text)
    query_vector = embed_text(job_text, task_type="RETRIEVAL_QUERY") if job_text.strip() else []

    rows: list[dict[str, Any]] = []
    source = "local"
    if user_id:
        try:
            if query_vector:
                vector_rows = xano.vector_search_chunks(query_vector, user_id, k=max(k, 12))
                if vector_rows is not None:
                    rows = vector_rows
                    source = "xano_vector"
            if not rows:
                rows = [row for row in xano.list_chunks(user_id) if _is_active(row)]
                source = "xano_list"
        except XanoError:
            logger.exception("Chunk retrieval could not read Xano")
            rows = []
            source = "local"

    if not rows:
        rows = [chunk for chunk in (local_chunks or []) if _is_active(chunk)]
        source = "local"

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        if not _is_active(row):
            continue
        chunk_vector = _as_vector(row.get("embedding"))
        embedding_score = (
            _dense_cosine(query_vector, chunk_vector) if query_vector and chunk_vector else 0.0
        )
        lexical = _lexical_score(row, job_text, job_skills)
        score = (0.7 * embedding_score + 0.3 * min(lexical / 6.0, 1.0)) if embedding_score else lexical
        scored.append((score, row))

    scored.sort(key=lambda item: -item[0])
    return [row for _, row in scored[:k]], source
