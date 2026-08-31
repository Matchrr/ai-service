"""Agent C: semantic retrieval over active Xano jobs, then lexical scorecard blend."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.db.xano import XanoError, xano
from app.pipelines.embeddings import embed_text
from app.pipelines.jobs import now_ms

logger = logging.getLogger(__name__)

_WORK_MODE = {
    "remote": "remote",
    "hybrid": "hybrid",
    "onsite": "onsite",
    "on-site": "onsite",
    "on_site": "onsite",
}

_JOB_TYPE = {
    "full_time": "full_time",
    "full-time": "full_time",
    "part_time": "part_time",
    "part-time": "part_time",
    "contract": "contract",
    "internship": "internship",
    "seasonal": "seasonal",
    "internship": "internship",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.]*")

# Weights for the combined semantic score used to rank retrieved jobs.
# The raw embedding cosine is the dominant signal; title and skill overlap
# provide cheap, interpretable corrections when the embedding is noisy.
_W_EMBEDDING = 0.55
_W_TITLE = 0.15
_W_SKILLS = 0.20
_W_WORK_MODE = 0.05
_W_LOCATION = 0.05


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


def _as_vector(value: Any) -> list[float]:
    if isinstance(value, list) and value and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    return []


def _norm(value: str | None) -> str:
    return (value or "").strip().lower().replace("_", "-")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _title_affinity(target_title: str | None, job_title: str | None) -> float:
    if not target_title or not job_title:
        return 0.0
    target_tokens = _tokenize(target_title)
    job_tokens = _tokenize(job_title)
    if not target_tokens or not job_tokens:
        return 0.0
    overlap = len(target_tokens & job_tokens)
    return overlap / len(target_tokens)


def _skill_overlap(candidate_skills: list[str], job_skills: list[str]) -> float:
    if not candidate_skills or not job_skills:
        return 0.0
    left = {skill.lower().strip() for skill in candidate_skills if skill}
    right = {skill.lower().strip() for skill in job_skills if skill}
    if not left or not right:
        return 0.0
    intersection = left & right
    union = left | right
    return len(intersection) / len(union)


def _work_mode_match(row: dict[str, Any], work_modes: list[str] | None) -> float:
    if not work_modes:
        return 0.0
    allowed = {_WORK_MODE.get(_norm(mode), _norm(mode)) for mode in work_modes}
    mode = _WORK_MODE.get(_norm(str(row.get("work_mode") or "")), _norm(str(row.get("work_mode") or "")))
    return 1.0 if mode and mode in allowed else 0.0


def _location_match(row: dict[str, Any], preferred_location: str | None) -> float:
    if not preferred_location:
        return 0.0
    job_location = str(row.get("location") or "").lower()
    preferred = preferred_location.lower().strip()
    if not job_location or "remote" in job_location:
        return 0.5
    if preferred in job_location:
        return 1.0
    # Check if any token of the preferred location (e.g. city or region) appears.
    preferred_tokens = _tokenize(preferred)
    job_tokens = _tokenize(job_location)
    if not preferred_tokens or not job_tokens:
        return 0.0
    overlap = len(preferred_tokens & job_tokens)
    return overlap / len(preferred_tokens)


def _build_query_text(
    profile_text: str,
    target_title: str | None,
    skills: list[str] | None,
    work_modes: list[str] | None,
    location: str | None,
) -> str:
    parts = [profile_text]
    if target_title:
        parts.append(f"Target role: {target_title}")
    if skills:
        parts.append(f"Skills: {', '.join(skills)}")
    if work_modes:
        parts.append(f"Preferred work arrangement: {', '.join(work_modes)}")
    if location:
        parts.append(f"Preferred location: {location}")
    return "\n".join(parts)


def _build_filters(
    cutoff: int,
    work_modes: list[str] | None,
    job_types: list[str] | None,
    pay_min: float | None,
    pay_period: str | None,
) -> dict[str, Any]:
    filters: dict[str, Any] = {
        "status": "active",
        "last_seen_at": {"gt": cutoff},
    }
    if work_modes:
        normalized = {_WORK_MODE.get(_norm(mode), _norm(mode)) for mode in work_modes}
        if normalized:
            filters["work_mode"] = {"in": list(normalized)}
    if job_types:
        normalized = {_JOB_TYPE.get(_norm(kind), _norm(kind)) for kind in job_types}
        if normalized:
            filters["job_type"] = {"in": list(normalized)}
    if pay_min and pay_min > 0:
        filters["pay_min"] = {"gte": pay_min}
        if pay_period:
            filters["pay_period"] = pay_period
    return filters


def _passes_filters(
    row: dict[str, Any],
    work_modes: list[str] | None,
    job_types: list[str] | None,
    pay_min: float | None,
    pay_period: str | None,
) -> bool:
    if work_modes:
        allowed = {_WORK_MODE.get(_norm(mode), _norm(mode)) for mode in work_modes}
        mode = _WORK_MODE.get(_norm(str(row.get("work_mode") or "")), _norm(str(row.get("work_mode") or "")))
        if mode and allowed and mode not in allowed:
            return False
    if job_types:
        allowed = {_JOB_TYPE.get(_norm(kind), _norm(kind)) for kind in job_types}
        kind = _JOB_TYPE.get(_norm(str(row.get("job_type") or "")), _norm(str(row.get("job_type") or "")))
        if kind and allowed and kind not in allowed:
            return False
    if pay_min and pay_min > 0:
        job_min = float(row.get("pay_min") or 0)
        job_max = float(row.get("pay_max") or 0)
        job_pay = job_max or job_min
        if job_pay:
            pay_type = _norm(str(row.get("pay_type") or ""))
            period = _norm(pay_period or "annual")
            if pay_type == "hourly" and period in {"annual", "salary"}:
                job_pay *= 2080
            elif pay_type in {"annual", "salary", ""} and period == "hourly":
                job_pay /= 2080
            if job_pay < pay_min:
                return False
    return True


def _posted_label(row: dict[str, Any]) -> str | None:
    posted = int(row.get("date_posted") or 0)
    if not posted:
        return None
    age = max(0, now_ms() - posted)
    days = age / (24 * 3600 * 1000)
    if days < 1:
        hours = max(1, round(age / (3600 * 1000)))
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    whole = max(1, round(days))
    return f"{whole} day{'s' if whole != 1 else ''} ago"


def _salary_label(row: dict[str, Any]) -> str | None:
    pay_min = float(row.get("pay_min") or 0)
    pay_max = float(row.get("pay_max") or 0)
    if not pay_min and not pay_max:
        return None
    pay_type = str(row.get("pay_type") or "annual")
    suffix = "/hr" if pay_type.lower() == "hourly" else ""

    def fmt(value: float) -> str:
        if pay_type.lower() == "hourly":
            return f"${value:,.0f}"
        if value >= 1000:
            return f"${value/1000:.0f}k"
        return f"${value:,.0f}"

    if pay_min and pay_max and pay_min != pay_max:
        return f"{fmt(pay_min)} – {fmt(pay_max)}{suffix}"
    return f"{fmt(pay_max or pay_min)}{suffix}"


def _to_match(row: dict[str, Any], similarity: float | None, semantic_score: float | None) -> dict[str, Any]:
    external_id = str(row.get("external_id") or row.get("id") or "")
    return {
        "id": external_id,
        "external_id": external_id,
        "xano_id": row.get("id"),
        "title": row.get("job_position_name") or "",
        "company": row.get("company_name") or "",
        "location": row.get("location") or None,
        "source": row.get("job_source") or None,
        "posted_at": _posted_label(row),
        "salary": _salary_label(row),
        "apply_url": row.get("apply_url") or row.get("source_url") or None,
        "description": row.get("job_description") or None,
        "similarity": None if similarity is None else round(similarity, 4),
        "semantic_score": None if semantic_score is None else round(semantic_score, 4),
        "work_mode": row.get("work_mode") or None,
        "job_type": row.get("job_type") or None,
        "pay_min": row.get("pay_min") or 0,
        "pay_max": row.get("pay_max") or 0,
        "pay_type": row.get("pay_type") or None,
    }


class SemanticMatchAgent:
    name = "semantic_match"

    def run(
        self,
        candidate_id: str,
        limit: int = 10,
        *,
        target_title: str | None = None,
        profile_text: str = "",
        skills: list[str] | None = None,
        work_modes: list[str] | None = None,
        job_types: list[str] | None = None,
        pay_min: float | None = None,
        pay_period: str | None = None,
        location: str | None = None,
        neighbor_count: int = 50,
    ) -> dict:
        cutoff = now_ms() - settings.harvest_stale_hours * 3600 * 1000
        filters = _build_filters(cutoff, work_modes, job_types, pay_min, pay_period)
        query_text = _build_query_text(profile_text, target_title, skills, work_modes, location)
        query_vector = embed_text(query_text, task_type="RETRIEVAL_QUERY") if query_text.strip() else []

        # 1. Try Xano vector search (custom function). Fall back to brute-force cosine.
        rows: list[dict[str, Any]] = []
        vector_source = "brute_force"
        try:
            if query_vector:
                vector_rows = xano.vector_search_jobs(query_vector, filters=filters, k=neighbor_count)
                if vector_rows is not None:
                    rows = vector_rows
                    vector_source = "xano_vector"
        except XanoError:
            logger.exception("Xano vector search failed; falling back to brute force")

        if not rows:
            try:
                rows = xano.list_jobs()
            except XanoError:
                logger.exception("Match could not list Xano jobs")
                return {
                    "agent": self.name,
                    "candidate_id": candidate_id,
                    "matches": [],
                    "limit": limit,
                    "source": "error",
                }

        # 2. Filter to active, recent, and user-selected constraints.
        pool: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("status") or "") != "active":
                continue
            last_seen = int(row.get("last_seen_at") or 0)
            if last_seen < cutoff:
                continue
            if not str(row.get("external_id") or "").strip():
                continue
            if not _passes_filters(row, work_modes, job_types, pay_min, pay_period):
                continue
            pool.append(row)

        # 3. Score and rank with embedding cosine + semantic signals.
        candidate_skills = skills or []
        scored: list[tuple[float, float, dict[str, Any]]] = []
        for row in pool:
            job_vector = _as_vector(row.get("embedding"))
            embedding_similarity = (
                _dense_cosine(query_vector, job_vector) if query_vector and job_vector else 0.0
            )

            title_sim = _title_affinity(target_title, row.get("job_position_name"))
            skill_sim = _skill_overlap(candidate_skills, row.get("required_skills") or [])
            mode_sim = _work_mode_match(row, work_modes)
            loc_sim = _location_match(row, location)

            semantic_score = (
                _W_EMBEDDING * embedding_similarity
                + _W_TITLE * title_sim
                + _W_SKILLS * skill_sim
                + _W_WORK_MODE * mode_sim
                + _W_LOCATION * loc_sim
            )
            scored.append((semantic_score, embedding_similarity, row))

        scored.sort(key=lambda item: -item[0])
        top = scored[: max(neighbor_count, limit)]
        matches = [_to_match(row, embedding_similarity, semantic_score) for semantic_score, embedding_similarity, row in top]
        return {
            "agent": self.name,
            "candidate_id": candidate_id,
            "target_title": target_title,
            "matches": matches,
            "limit": limit,
            "pool_size": len(pool),
            "source": vector_source,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
