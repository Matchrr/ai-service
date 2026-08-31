"""Compose a grounded cover letter from retrieved profile chunks and persist it."""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.db.xano import XanoError, xano
from app.lib.skills import SKILL_ALIASES, extract_skills
from app.pipelines.embeddings import embed_text
from app.pipelines.jobs import content_hash
from app.pipelines.profile_chunks import facts_to_chunks, retrieve_chunks, upsert_profile_chunks

logger = logging.getLogger(__name__)

GENERATION_RAG = "rag"
GENERATION_FALLBACK = "fallback_template"
TONE_PROFESSIONAL = "professional"
STATUS_GENERATED = "generated"
STATUS_DISCARDED = "discarded"
GENERATION_MODEL = "grounded-template"

_NEGATION_MARKERS = (
    "have not",
    "haven't",
    "not in production",
    "do not have",
    "don't have",
    "would not claim",
    "actively closing",
    "closing that gap",
    "not claim",
)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _affirmative_text(text: str) -> str:
    sentences = [part.strip() for part in text.replace("\n", " ").split(".") if part.strip()]
    kept = [
        sentence
        for sentence in sentences
        if not any(marker in sentence.lower() for marker in _NEGATION_MARKERS)
    ]
    return ". ".join(kept)


def verify_grounding(profile: dict[str, Any], texts: list[str]) -> dict[str, Any]:
    owned = {skill.lower() for skill in (profile.get("skills") or [])}
    source_parts = [str(profile.get("summary") or "")]
    for role in list(profile.get("experience") or []) + list(profile.get("volunteering") or []):
        if not isinstance(role, dict):
            continue
        source_parts.append(str(role.get("title") or ""))
        source_parts.append(str(role.get("company") or ""))
        source_parts.extend(str(item) for item in (role.get("bullets") or []))
    source_parts.extend(str(item) for item in (profile.get("projects") or []))
    source_text = " ".join(source_parts).lower()

    rejected: list[str] = []
    checked = 0
    for text in texts:
        for skill in extract_skills(text):
            checked += 1
            if skill.lower() in owned:
                continue
            aliases = [skill.lower(), *(SKILL_ALIASES.get(skill) or [])]
            if any(alias in source_text for alias in aliases):
                continue
            rejected.append(skill)
    unique = sorted(set(rejected))
    passed = not unique
    note = (
        f"All {checked} technical claims trace back to your verified history."
        if passed
        else f"Blocked {len(unique)} unverified claim(s): {', '.join(unique)}."
    )
    return {
        "passed": passed,
        "claims_checked": checked,
        "rejected_claims": unique,
        "note": note,
    }


def _current_role(profile: dict[str, Any]) -> tuple[str, str]:
    for role in profile.get("experience") or []:
        if isinstance(role, dict) and (role.get("title") or role.get("company")):
            return str(role.get("title") or "").strip(), str(role.get("company") or "").strip()
    return "", ""


def compose_cover_letter(
    profile: dict[str, Any],
    job_title: str,
    company: str,
    chunks: list[dict[str, Any]],
    matching_skills: list[str],
    missing_tech: list[str],
) -> tuple[str, list[str]]:
    name = str(profile.get("full_name") or "your candidate").strip() or "your candidate"
    title, current_company = _current_role(profile)
    current_line = (
        f"I am currently a {title} at {current_company}"
        if title
        else "I am currently between roles"
    )
    highlighted = list(matching_skills[:3]) if matching_skills else []
    if not highlighted:
        for chunk in chunks:
            for skill in chunk.get("skills") or []:
                if skill not in highlighted:
                    highlighted.append(skill)
                if len(highlighted) >= 3:
                    break
            if len(highlighted) >= 3:
                break
    skills_line = ", ".join(highlighted[:3]) if highlighted else "the fundamentals"

    evidence: list[str] = []
    for chunk in chunks:
        text = str(chunk.get("chunk_text") or "").strip()
        if text and text not in evidence:
            evidence.append(text)
        if len(evidence) >= 3:
            break
    if not evidence:
        return "", highlighted
    evidence_block = "\n".join(f"- {item}" for item in evidence)

    gap = ""
    if missing_tech:
        first = missing_tech[0]
        gap = (
            f"\n\nI have not shipped {first} in production. I am working through it now, and I "
            f"would rather tell you that up front than discover it in week one."
        )

    greeting = f"Dear {company} Hiring Team" if company else "Dear Hiring Team"
    letter = (
        f"{greeting},\n\n"
        f"I am applying for the {job_title} role. {current_line}, where my work centers on "
        f"{skills_line} — the same ground your posting describes.\n\n"
        f"What from my record maps directly to this role:\n{evidence_block}\n\n"
        f"What draws me to {company} specifically is that the problem is infrastructural "
        f"rather than cosmetic: the posting asks for judgment about correctness and scale, "
        f"which is where I have spent my career."
        f"{gap}\n\n"
        f"Thank you for your time.\n\n{name}"
    )
    return letter, highlighted[:8]


def _next_version(user_id: int, xano_job_id: int | None) -> tuple[int, int | None]:
    try:
        rows = xano.list_cover_letters(user_id)
    except XanoError:
        return 1, None
    related = [
        row
        for row in rows
        if xano_job_id is None or _as_int(row.get("matchrr_job_position_id")) == xano_job_id
    ]
    if not related:
        return 1, None
    related.sort(key=lambda row: int(row.get("version") or 0))
    latest = related[-1]
    return int(latest.get("version") or 0) + 1, _as_int(latest.get("id"))


def _letter_payload(
    *,
    user_id: int,
    xano_job_id: int | None,
    letter: str,
    company: str,
    job_title: str,
    source: str,
    highlighted: list[str],
    status: str,
    version: int,
    parent_id: int | None,
    job_snapshot: dict[str, Any],
    profile_revision_id: int | None,
    grounding_passed: bool,
    rejected_claims: list[str],
    retrieved_chunk_ids: list[int],
    application_id: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_id": user_id,
        "cover_letter_text": letter,
        "company_name": company,
        "job_title": job_title,
        "generation_source": source,
        "generation_model": GENERATION_MODEL,
        "tone": TONE_PROFESSIONAL,
        "highlighted_skills": highlighted,
        "status": status,
        "version": version,
        "content_hash": content_hash(letter),
        "job_snapshot": job_snapshot,
        "grounding_passed": grounding_passed,
        "rejected_claims": rejected_claims,
        "retrieved_chunk_ids": retrieved_chunk_ids,
    }
    if xano_job_id:
        payload["matchrr_job_position_id"] = xano_job_id
    if parent_id:
        payload["parent_cover_letter_id"] = parent_id
    if profile_revision_id:
        payload["profile_revision_id"] = profile_revision_id
    if application_id:
        payload["application_id"] = application_id
    vector = embed_text(letter, task_type="RETRIEVAL_DOCUMENT") if letter.strip() else []
    expected = settings.embedding_dimensions or 1536
    if vector and len(vector) == expected:
        payload["embedding"] = vector
    return payload


def persist_cover_letter(payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return xano.add_cover_letter(payload)
    except XanoError as exc:
        # Screenshot typo some tables still have: retreived_chunk_ids
        if "retrieved_chunk_ids" in payload:
            retry = dict(payload)
            retry["retreived_chunk_ids"] = retry.pop("retrieved_chunk_ids")
            try:
                return xano.add_cover_letter(retry)
            except XanoError:
                logger.exception("Could not persist cover letter: %s", exc)
                return None
        logger.exception("Could not persist cover letter: %s", exc)
        return None


def generate_cover_letter(payload: dict[str, Any]) -> dict[str, Any]:
    profile = payload.get("profile") or {}
    user_id = _as_int(payload.get("user_id"))
    job_title = str(payload.get("job_title") or "").strip()
    company = str(payload.get("company") or "").strip()
    job_description = str(payload.get("job_description") or "")
    matching_skills = list(payload.get("matching_skills") or [])
    missing_tech = list(payload.get("missing_tech") or [])
    job_text = f"{job_title} at {company}\n{job_description}\n{' '.join(matching_skills)}"
    local_chunks = facts_to_chunks(profile)

    chunk_stats: dict[str, Any] = {}
    if user_id:
        try:
            chunk_stats = upsert_profile_chunks(user_id, profile)
        except XanoError:
            logger.exception("Cover letter generate could not upsert chunks")
            chunk_stats = {"error": "upsert_failed"}

    chunks, retrieve_source = retrieve_chunks(user_id, job_text, local_chunks=local_chunks)
    letter, highlighted = compose_cover_letter(
        profile, job_title, company, chunks, matching_skills, missing_tech
    )
    source = GENERATION_RAG if chunks and letter else GENERATION_FALLBACK
    if not letter:
        letter, highlighted = compose_cover_letter(
            profile, job_title, company, local_chunks[:3], matching_skills, missing_tech
        )
        source = GENERATION_FALLBACK

    grounding = verify_grounding(profile, [_affirmative_text(letter)])
    if not grounding["passed"] and source == GENERATION_RAG:
        letter, highlighted = compose_cover_letter(
            profile, job_title, company, local_chunks[:3], matching_skills, missing_tech
        )
        source = GENERATION_FALLBACK
        grounding = verify_grounding(profile, [_affirmative_text(letter)])

    retrieved_ids = [rid for rid in (_as_int(chunk.get("id")) for chunk in chunks) if rid]
    status = STATUS_GENERATED if grounding["passed"] else STATUS_DISCARDED
    job_snapshot = {
        "title": job_title,
        "company": company,
        "description_excerpt": job_description[:1200],
        "key_angle": payload.get("key_angle"),
        "matching_skills": matching_skills,
    }

    stored: dict[str, Any] | None = None
    if user_id and letter:
        xano_job_id = _as_int(payload.get("xano_job_id"))
        version, parent_id = _next_version(user_id, xano_job_id)
        stored = persist_cover_letter(
            _letter_payload(
                user_id=user_id,
                xano_job_id=xano_job_id,
                letter=letter,
                company=company,
                job_title=job_title,
                source=source,
                highlighted=highlighted,
                status=status,
                version=version,
                parent_id=parent_id,
                job_snapshot=job_snapshot,
                profile_revision_id=_as_int(payload.get("profile_revision_id")),
                grounding_passed=bool(grounding["passed"]),
                rejected_claims=list(grounding["rejected_claims"]),
                retrieved_chunk_ids=retrieved_ids,
                application_id=_as_int(payload.get("application_id")),
            )
        )

    return {
        "cover_letter": letter,
        "highlighted_skills": highlighted,
        "retrieved_chunk_ids": retrieved_ids,
        "retrieved_chunks": [
            {
                "id": chunk.get("id"),
                "source_type": chunk.get("source_type"),
                "chunk_text": chunk.get("chunk_text"),
                "company": chunk.get("company") or "",
            }
            for chunk in chunks
        ],
        "generation_source": source,
        "retrieve_source": retrieve_source,
        "grounding": grounding,
        "chunk_stats": chunk_stats,
        "cover_letter_id": stored.get("id") if stored else None,
        "persisted": stored is not None,
    }
