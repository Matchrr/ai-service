"""SerpAPI Google Jobs harvest → embed → upsert to Xano."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import settings
from app.db.xano import XanoError, xano
from app.lib.skills import extract_skills
from app.pipelines.embeddings import embed_texts
from app.pipelines.queries import CHIPS_WEEK, SearchQuery, allocate_standing_queries, expand_queries

logger = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search.json"
AGGREGATORS = (
    "indeed",
    "linkedin",
    "ziprecruiter",
    "glassdoor",
    "simplyhired",
    "monster",
    "dice.com",
    "google.com",
)

# Scraped / expired-heavy boards. Prefer dropping these over storing dead apply links.
JUNK_APPLY_HOSTS = (
    "bebee.com",
    "jobleads.com",
    "jobright.ai",
    "learn4good.com",
    "jobtarget.com",
    "bandana.com",
    "jobgether.com",
    "jobrapido.com",
    "haystackapp.io",
    "dejobs.org",
    "jobisjob.com",
    "neuvoo.com",
    "talent.com",
    "jooble.org",
    "adzuna.com",
    "careerjet.com",
    "sk-improvement.com",
    "lensa.com",
    "anitab.org",
    "google.com",
    "talents.vaia.com",
    "jobilize.com",
    "trabajo.org",
    "recruit.net",
    "jobmesh.io",
    "lifeworq.com",
)

# Xano enum values on matchrr_job_position (lowercase). Title-case is rejected.
XANO_WORK_MODE = {
    "remote": "remote",
    "hybrid": "hybrid",
    "onsite": "onsite",
    "on-site": "onsite",
    "on_site": "onsite",
    "on site": "onsite",
    "in-office": "onsite",
    "in office": "onsite",
    "office": "onsite",
    "in-person": "onsite",
    "presencial": "onsite",
    "remoto": "remote",
    "híbrido": "hybrid",
    "hibrido": "hybrid",
    "wfh": "remote",
    "work from home": "remote",
}

XANO_JOB_TYPE = {
    "full_time": "full_time",
    "full-time": "full_time",
    "full time": "full_time",
    "fulltime": "full_time",
    "tiempo completo": "full_time",
    "part_time": "part_time",
    "part-time": "part_time",
    "part time": "part_time",
    "tiempo parcial": "part_time",
    "contract": "contract",
    "contractor": "contract",
    "contratista": "contract",
    "temporary": "contract",
    "temp": "contract",
    "freelance": "contract",
    "internship": "internship",
    "intern": "internship",
    "seasonal": "seasonal",
    "season": "seasonal",
}

XANO_PAY_TYPE = {
    "hourly": "hourly",
    "hour": "hourly",
    "/hr": "hourly",
    "por hora": "hourly",
    "annual": "annual",
    "annually": "annual",
    "salary": "annual",
    "yearly": "annual",
    "year": "annual",
    "al año": "annual",
    "al ano": "annual",
}

MAX_INGEST_AGE_DAYS = 7

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([kmb])?", re.I)
_RELATIVE_RE = re.compile(
    r"(?:(\d+)\s+)?(minute|hour|day|week|month|year)s?\s+ago|just posted|today|yesterday",
    re.I,
)
_ES_RELATIVE_RE = re.compile(
    r"hace\s+(?:(\d+)\s+)?(minuto|hora|d[ií]a|semana|mes|a[nñ]o)s?|hoy|ayer",
    re.I,
)

_memory_runs: dict[tuple[str, str, str], dict[str, Any]] = {}


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug(text: str) -> str:
    return _SLUG_RE.sub("-", (text or "").lower()).strip("-")


def fingerprint_job(company: str, title: str, location: str) -> str:
    city = (location or "").split(",")[0].strip()
    return f"{slug(company)}|{slug(title)}|{slug(city)}"


def content_hash(description: str) -> str:
    normalized = re.sub(r"\s+", " ", (description or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _map_enum(raw: str | None, table: dict[str, str], default: str = "") -> str:
    text = (raw or "").strip().lower()
    if not text:
        return default
    if text in table:
        return table[text]
    collapsed = text.replace("-", " ").replace("_", " ")
    if collapsed in table:
        return table[collapsed]
    underscored = collapsed.replace(" ", "_")
    if underscored in table:
        return table[underscored]
    for key, value in sorted(table.items(), key=lambda item: -len(item[0])):
        if len(key) >= 5 and key in text:
            return value
    return default


def parse_relative_posted_at(raw: str | None) -> int | None:
    if not raw:
        return None
    text = raw.strip().lower()
    now = datetime.now(timezone.utc)
    if "just" in text or text in {"today", "hoy"}:
        return int(now.timestamp() * 1000)
    if text in {"yesterday", "ayer"}:
        return int((now - timedelta(days=1)).timestamp() * 1000)
    match = _RELATIVE_RE.search(text)
    amount = 1
    unit = ""
    if match:
        amount = int(match.group(1) or 1)
        unit = (match.group(2) or "day").lower()
    else:
        es = _ES_RELATIVE_RE.search(text)
        if not es:
            return None
        if es.group(0).lower() in {"hoy", "ayer"}:
            days = 0 if es.group(0).lower() == "hoy" else 1
            return int((now - timedelta(days=days)).timestamp() * 1000)
        amount = int(es.group(1) or 1)
        unit = (es.group(2) or "día").lower()
        unit = {
            "minuto": "minute",
            "hora": "hour",
            "día": "day",
            "dia": "day",
            "semana": "week",
            "mes": "month",
            "año": "year",
            "ano": "year",
        }.get(unit, unit)
    delta = {
        "minute": timedelta(minutes=amount),
        "hour": timedelta(hours=amount),
        "day": timedelta(days=amount),
        "week": timedelta(weeks=amount),
        "month": timedelta(days=30 * amount),
        "year": timedelta(days=365 * amount),
    }.get(unit)
    if delta is None:
        return None
    return int((now - delta).timestamp() * 1000)


def _parse_money_token(token: str) -> float | None:
    match = _NUMBER_RE.search(token.replace(",", ""))
    if not match:
        return None
    value = float(match.group(1))
    suffix = (match.group(2) or "").lower()
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    elif suffix == "b":
        value *= 1_000_000_000
    return value


def parse_salary(raw: str | None) -> tuple[float | None, float | None, str]:
    if not raw:
        return None, None, ""
    text = raw.lower()
    if any(word in text for word in ("hour", "/hr", "hourly", "por hora")):
        pay_type = "hourly"
    elif any(word in text for word in ("year", "annual", "salary", "al año", "al ano", "/yr")):
        pay_type = "annual"
    else:
        pay_type = ""
    amounts = [_parse_money_token(part) for part in re.split(r"[-–—to]+", raw)]
    amounts = [amount for amount in amounts if amount]
    if not pay_type and amounts:
        peak = max(amounts)
        pay_type = "hourly" if peak < 500 else "annual"
    if not amounts:
        return None, None, _map_enum(pay_type, XANO_PAY_TYPE)
    return min(amounts), max(amounts), _map_enum(pay_type, XANO_PAY_TYPE, "annual")


def _host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _is_aggregator(url: str, title: str = "") -> bool:
    blob = f"{url} {title}".lower()
    host = _host(url)
    return any(name in blob or name in host for name in AGGREGATORS)


def _is_junk_apply(url: str, title: str = "") -> bool:
    if not url:
        return True
    blob = f"{url} {title}".lower()
    host = _host(url)
    return any(name in host or name in blob for name in JUNK_APPLY_HOSTS)


def pick_apply_url(raw: dict[str, Any]) -> str:
    options = raw.get("apply_options") or []
    links = [(str(item.get("link") or ""), str(item.get("title") or "")) for item in options if item]
    company = [
        link
        for link, title in links
        if link and not _is_aggregator(link, title) and not _is_junk_apply(link, title)
    ]
    if company:
        return company[0]
    decent = [link for link, title in links if link and not _is_junk_apply(link, title)]
    if decent:
        return decent[0]
    related = raw.get("related_links") or []
    for item in related:
        link = str(item.get("link") or "")
        if link and not _is_junk_apply(link):
            return link
    share = str(raw.get("share_link") or raw.get("link") or "")
    if share and not _is_junk_apply(share):
        return share
    return ""


def infer_work_mode(location: str, extensions: dict[str, Any], description: str, title: str = "") -> str:
    blob = " ".join(
        [
            title or "",
            location or "",
            str(extensions.get("work_from_home") or ""),
            str(extensions.get("schedule_type") or ""),
            str(extensions.get("work_mode") or ""),
            description[:800],
        ]
    ).lower()
    if "hybrid" in blob or "híbrido" in blob or "hibrido" in blob:
        return "hybrid"
    if (
        "remote" in blob
        or "remoto" in blob
        or "wfh" in blob
        or "work from home" in blob
        or extensions.get("work_from_home")
    ):
        return "remote"
    return "onsite"


def infer_job_type(schedule: str) -> str:
    mapped = _map_enum(schedule, XANO_JOB_TYPE)
    if mapped:
        return mapped
    text = (schedule or "").lower()
    if "intern" in text:
        return "internship"
    if "season" in text:
        return "seasonal"
    if "part" in text:
        return "part_time"
    if "contract" in text or "temp" in text or "freelance" in text or "contratista" in text:
        return "contract"
    return "full_time"


def listing_is_fresh(job: dict[str, Any]) -> bool:
    """Drop listings Google still surfaces that are clearly older than the week chip."""
    posted_raw = str(job.get("_posted_raw") or "").lower()
    if any(token in posted_raw for token in ("month", "year", "mes", "año", "ano")):
        return False
    posted = int(job.get("date_posted") or 0)
    if not posted:
        return True
    age_days = (now_ms() - posted) / (24 * 3600 * 1000)
    return age_days <= MAX_INGEST_AGE_DAYS


def _valid_embedding(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    vector = row.get("embedding")
    if not isinstance(vector, list) or not vector:
        return False
    expected = settings.embedding_dimensions or 1536
    return len(vector) == expected and all(isinstance(item, (int, float)) for item in vector[:8])


def _highlights_blob(raw: dict[str, Any]) -> tuple[str, list[str], str]:
    sections = raw.get("job_highlights") or []
    chunks: list[str] = []
    benefits: list[str] = []
    qualifications: list[str] = []
    for section in sections:
        title = str(section.get("title") or "")
        items = [str(item) for item in (section.get("items") or []) if item]
        if not items:
            continue
        chunks.append(f"{title}: " + " ".join(items) if title else " ".join(items))
        lowered = title.lower()
        if "benefit" in lowered:
            benefits.extend(items)
        if "qualif" in lowered or "responsib" in lowered:
            qualifications.extend(items)
    return "\n".join(chunks), benefits, " ".join(qualifications)


def normalize_job(raw: dict[str, Any]) -> dict[str, Any]:
    title = str(raw.get("title") or "").strip()
    company = str(raw.get("company_name") or raw.get("company") or "").strip()
    location = str(raw.get("location") or "").strip()
    highlights, benefits, qualifications = _highlights_blob(raw)
    description = str(raw.get("description") or "").strip()
    if highlights:
        description = f"{description}\n{highlights}".strip()
    extensions = raw.get("detected_extensions") or {}
    posted_raw = str(extensions.get("posted_at") or "")
    schedule = str(extensions.get("schedule_type") or "")
    salary_raw = str(extensions.get("salary") or "")
    pay_min, pay_max, pay_type = parse_salary(salary_raw)
    apply_url = pick_apply_url(raw)
    experience = str(extensions.get("qualifications") or "")
    skills = extract_skills(f"{description} {qualifications}")
    return {
        "job_position_name": title,
        "job_description": description,
        "job_source": str(raw.get("via") or ""),
        "company_name": company,
        "location": location,
        "pay_min": pay_min or 0,
        "pay_max": pay_max or 0,
        "benefits": benefits,
        "schedule": schedule,
        "experience_level": experience,
        "required_skills": skills,
        "date_posted": parse_relative_posted_at(posted_raw) or 0,
        "source_url": apply_url,
        "external_id": str(raw.get("job_id") or "").strip(),
        "status": "active",
        "content_hash": content_hash(description),
        "fingerprint": fingerprint_job(company, title, location),
        "last_seen_at": now_ms(),
        "apply_url": apply_url,
        "work_mode": infer_work_mode(location, extensions, description, title),
        "job_type": infer_job_type(schedule),
        "pay_type": pay_type,
        "_posted_raw": posted_raw,
        "_salary_raw": salary_raw,
        "_highlights": qualifications,
    }


def embed_text_for_job(job: dict[str, Any]) -> str:
    return (
        f"{job.get('job_position_name') or ''} at {job.get('company_name') or ''} "
        f"in {job.get('location') or ''}\n"
        f"{job.get('job_description') or ''}\n"
        f"Qualifications: {job.get('_highlights') or ''}"
    )


def fetch_google_jobs(q: str, location: str | None = None) -> tuple[list[dict[str, Any]], str, str | None]:
    """Fetch up to HARVEST_PAGES of Google Jobs. Returns (raw jobs, serpapi_search_id, warning)."""
    if not settings.serpapi_api_key:
        raise RuntimeError("SERPAPI_API_KEY is not set")

    jobs: list[dict[str, Any]] = []
    search_id = ""
    warning: str | None = None
    token: str | None = None

    for page in range(max(1, settings.harvest_pages)):
        params: dict[str, Any] = {
            "engine": "google_jobs",
            "q": q,
            "chips": CHIPS_WEEK,
            "hl": "en",
            "api_key": settings.serpapi_api_key,
        }
        if location:
            params["location"] = location
        if token:
            params["next_page_token"] = token

        payload = _serpapi_get(params)
        if payload is None:
            warning = "SerpAPI request failed"
            break
        search_id = str((payload.get("search_metadata") or {}).get("id") or search_id)
        results = payload.get("jobs_results") or []
        if not results and page == 0:
            time.sleep(2)
            payload = _serpapi_get(params)
            if payload is None:
                warning = "SerpAPI empty after retry"
                break
            search_id = str((payload.get("search_metadata") or {}).get("id") or search_id)
            results = payload.get("jobs_results") or []
            if not results:
                warning = "SerpAPI empty after retry"
                break
        jobs.extend(item for item in results if isinstance(item, dict))
        token = (payload.get("serpapi_pagination") or {}).get("next_page_token")
        if not token:
            break

    return jobs, search_id, warning


def _serpapi_get(params: dict[str, Any], attempt: int = 0) -> dict[str, Any] | None:
    try:
        response = httpx.get(SERPAPI_URL, params=params, timeout=40.0)
    except httpx.HTTPError:
        logger.exception("SerpAPI network error")
        return None
    if response.status_code == 429 and attempt < 4:
        delay = 2 ** attempt
        logger.warning("SerpAPI 429; backing off %ss", delay)
        time.sleep(delay)
        return _serpapi_get(params, attempt + 1)
    if response.status_code >= 400:
        logger.warning("SerpAPI HTTP %s: %s", response.status_code, response.text[:300])
        return None
    try:
        return response.json()
    except ValueError:
        return None


def _index_catalog() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_external: dict[str, dict[str, Any]] = {}
    by_fingerprint: dict[str, dict[str, Any]] = {}
    try:
        rows = xano.list_jobs()
    except XanoError:
        logger.exception("Could not list Xano jobs")
        return by_external, by_fingerprint
    for row in rows:
        external_id = str(row.get("external_id") or "").strip()
        fingerprint = str(row.get("fingerprint") or "").strip()
        if external_id:
            by_external[external_id] = row
        if fingerprint:
            by_fingerprint.setdefault(fingerprint, row)
    return by_external, by_fingerprint


def _payload_for_xano(job: dict[str, Any], embedding: list[float] | None = None) -> dict[str, Any]:
    work_mode = _map_enum(str(job.get("work_mode") or ""), XANO_WORK_MODE, "onsite")
    job_type = _map_enum(str(job.get("job_type") or ""), XANO_JOB_TYPE, "full_time")
    pay_type = _map_enum(str(job.get("pay_type") or ""), XANO_PAY_TYPE)
    payload = {
        "job_position_name": job.get("job_position_name") or "",
        "job_description": job.get("job_description") or "",
        "job_source": job.get("job_source") or "",
        "company_name": job.get("company_name") or "",
        "location": job.get("location") or "",
        "pay_min": job.get("pay_min") or 0,
        "pay_max": job.get("pay_max") or 0,
        "benefits": job.get("benefits") or [],
        "schedule": job.get("schedule") or "",
        "experience_level": job.get("experience_level") or "",
        "required_skills": job.get("required_skills") or [],
        "date_posted": job.get("date_posted") or 0,
        "source_url": job.get("source_url") or "",
        "external_id": job.get("external_id") or "",
        "status": "active",
        "content_hash": job.get("content_hash") or "",
        "fingerprint": job.get("fingerprint") or "",
        "last_seen_at": job.get("last_seen_at") or now_ms(),
        "apply_url": job.get("apply_url") or "",
        "work_mode": work_mode,
        "job_type": job_type,
    }
    if pay_type:
        payload["pay_type"] = pay_type
    expected = settings.embedding_dimensions or 1536
    if embedding and len(embedding) == expected:
        payload["embedding"] = embedding
    return payload


def upsert_jobs(raw_jobs: list[dict[str, Any]]) -> dict[str, int]:
    inserted = updated = embedded = 0
    skipped_stale = skipped_junk = 0
    by_external, by_fingerprint = _index_catalog()
    to_embed: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []

    for raw in raw_jobs:
        job = normalize_job(raw)
        external_id = job["external_id"]
        if not external_id or not job["job_position_name"]:
            continue
        if not listing_is_fresh(job):
            skipped_stale += 1
            continue
        if _is_junk_apply(str(job.get("apply_url") or "")):
            skipped_junk += 1
            continue
        fingerprint = job["fingerprint"]
        existing = by_external.get(external_id)
        if existing is None and fingerprint:
            fp_hit = by_fingerprint.get(fingerprint)
            if fp_hit and str(fp_hit.get("external_id") or "") != external_id:
                existing = fp_hit

        if existing is None:
            to_embed.append(("insert", job, None))
            continue

        same_hash = str(existing.get("content_hash") or "") == job["content_hash"]
        missing_vector = not _valid_embedding(existing)
        if same_hash and not missing_vector:
            patch = {
                "last_seen_at": now_ms(),
                "status": "active",
                "apply_url": _merge_apply(existing.get("apply_url"), job.get("apply_url")),
                "source_url": existing.get("source_url") or job.get("source_url") or "",
                "location": job.get("location") or existing.get("location") or "",
                "job_source": job.get("job_source") or existing.get("job_source") or "",
                "work_mode": _map_enum(str(job.get("work_mode") or ""), XANO_WORK_MODE, "onsite"),
                "job_type": _map_enum(str(job.get("job_type") or ""), XANO_JOB_TYPE, "full_time"),
            }
            pay_type = _map_enum(str(job.get("pay_type") or ""), XANO_PAY_TYPE)
            if pay_type:
                patch["pay_type"] = pay_type
            try:
                updated_row = xano.patch_job(existing["id"], patch)
            except XanoError:
                logger.exception("Failed to touch job %s", existing.get("id"))
                continue
            updated += 1
            _remember(by_external, by_fingerprint, updated_row, job, existing)
            continue

        to_embed.append(("update", job, existing))

    texts = [embed_text_for_job(job) for _, job, _ in to_embed]
    vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT") if texts else []

    for (action, job, existing), vector in zip(to_embed, vectors):
        payload = _payload_for_xano(job, vector or None)
        try:
            row = _write_job(action, existing, payload)
        except XanoError:
            logger.exception("Failed to upsert %s", job.get("external_id"))
            continue
        if vector:
            embedded += 1
        if action == "insert":
            inserted += 1
        else:
            updated += 1
        _remember(by_external, by_fingerprint, row, job, existing)

    return {
        "inserted": inserted,
        "updated": updated,
        "embedded": embedded,
        "seen": len(raw_jobs),
        "skipped_stale": skipped_stale,
        "skipped_junk": skipped_junk,
    }


def _write_job(action: str, existing: dict[str, Any] | None, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    if action != "insert":
        body.pop("external_id", None)
    last_error: XanoError | None = None
    for _ in range(8):
        try:
            if action == "insert":
                return xano.add_job(body)
            assert existing is not None
            return xano.patch_job(existing["id"], body)
        except XanoError as exc:
            last_error = exc
            param = _input_error_param(str(exc))
            if param == "embedding" and "embedding" in body:
                logger.warning("Xano rejected embedding (%s); storing job without vector", exc)
                body.pop("embedding", None)
                continue
            if param and param in body:
                if param in {"work_mode", "job_type", "pay_type", "status"}:
                    logger.warning(
                        "Xano rejected enum %s=%r (%s); dropping field",
                        param,
                        body.get(param),
                        exc,
                    )
                body.pop(param, None)
                continue
            raise
    if last_error:
        raise last_error
    raise XanoError("Failed to write job")


def _input_error_param(message: str) -> str | None:
    match = re.search(r'"param"\s*:\s*"([^"]+)"', message)
    if match:
        return match.group(1)
    match = re.search(r"Input .+ is not one of the allowable values.+param\\?\":\\?\"([^\"]+)", message)
    return match.group(1) if match else None


def _merge_apply(existing: Any, incoming: Any) -> str:
    left = str(existing or "").strip()
    right = str(incoming or "").strip()
    if right and _is_junk_apply(right):
        right = ""
    if left and _is_junk_apply(left):
        left = ""
    if not left:
        return right
    if not right or right == left:
        return left
    if _is_aggregator(left) and not _is_aggregator(right):
        return right
    return left


def _remember(
    by_external: dict[str, dict[str, Any]],
    by_fingerprint: dict[str, dict[str, Any]],
    row: dict[str, Any],
    job: dict[str, Any],
    existing: dict[str, Any] | None,
) -> None:
    merged = {**(existing or {}), **job, **row}
    external_id = str(merged.get("external_id") or "")
    fingerprint = str(merged.get("fingerprint") or "")
    if external_id:
        by_external[external_id] = merged
    if fingerprint:
        by_fingerprint[fingerprint] = merged


def _run_key(q: str, location: str | None, chips: str) -> tuple[str, str, str]:
    return (q.strip().lower(), (location or "").strip().lower(), chips)


def _load_runs() -> dict[tuple[str, str, str], dict[str, Any]]:
    runs = dict(_memory_runs)
    for row in xano.list_runs():
        key = _run_key(str(row.get("q") or ""), row.get("location") or "", str(row.get("chips") or CHIPS_WEEK))
        runs[key] = row
    return runs


def _cache_hit(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    if str(row.get("error") or "").strip():
        return False
    cached_until = int(row.get("cached_until") or 0)
    return cached_until > now_ms()


def _write_run(
    query: SearchQuery,
    mode: str,
    cache_hours: int,
    result_count: int,
    search_id: str,
    error: str,
    existing: dict[str, Any] | None,
) -> None:
    payload = {
        "q": query.q,
        "location": query.location or "",
        "chips": CHIPS_WEEK,
        "mode": mode,
        "serpapi_search_id": search_id,
        "result_count": result_count,
        "error": error,
        "cached_until": now_ms() + cache_hours * 3600 * 1000,
    }
    key = _run_key(query.q, query.location, CHIPS_WEEK)
    _memory_runs[key] = payload
    record_id = (existing or {}).get("id")
    if record_id:
        xano.patch_run(record_id, payload)
    else:
        xano.add_run(payload)


def harvest_query(query: SearchQuery, mode: str, cache_hours: int) -> dict[str, Any]:
    raw, search_id, warning = fetch_google_jobs(query.q, query.location)
    stats = upsert_jobs(raw) if raw else {
        "inserted": 0,
        "updated": 0,
        "embedded": 0,
        "seen": 0,
        "skipped_stale": 0,
        "skipped_junk": 0,
    }
    stats.update(
        {
            "q": query.q,
            "location": query.location,
            "warning": warning,
            "search_id": search_id,
            "result_count": len(raw),
        }
    )
    return stats


def harvest_queries(queries: list[SearchQuery], mode: str, cache_hours: int) -> dict[str, Any]:
    runs = _load_runs()
    harvested = cached = 0
    inserted = updated = embedded = 0
    skipped_stale = skipped_junk = 0
    warnings: list[str] = []
    catalog_empty = not any(str(row.get("external_id") or "").strip() for row in xano.list_jobs())
    for query in queries:
        key = _run_key(query.q, query.location, CHIPS_WEEK)
        existing = runs.get(key)
        if _cache_hit(existing) and not catalog_empty:
            cached += 1
            continue
        stats = harvest_query(query, mode, cache_hours)
        harvested += 1
        inserted += int(stats.get("inserted") or 0)
        updated += int(stats.get("updated") or 0)
        embedded += int(stats.get("embedded") or 0)
        skipped_stale += int(stats.get("skipped_stale") or 0)
        skipped_junk += int(stats.get("skipped_junk") or 0)
        query_upserted = int(stats.get("inserted") or 0) + int(stats.get("updated") or 0)
        warning = str(stats.get("warning") or "")
        if warning:
            warnings.append(warning)
        hard_error = warning if warning and "empty" not in warning.lower() else ""
        if int(stats.get("result_count") or 0) > 0 and query_upserted == 0:
            skipped = int(stats.get("skipped_stale") or 0) + int(stats.get("skipped_junk") or 0)
            if skipped == 0:
                hard_error = hard_error or "upsert_failed"
        _write_run(
            query,
            mode,
            cache_hours,
            int(stats.get("result_count") or 0),
            str(stats.get("search_id") or ""),
            hard_error,
            existing,
        )
        runs[key] = {"cached_until": now_ms() + cache_hours * 3600 * 1000, "error": stats.get("warning") or ""}
    return {
        "synced": inserted + updated,
        "inserted": inserted,
        "updated": updated,
        "embedded": embedded,
        "skipped_stale": skipped_stale,
        "skipped_junk": skipped_junk,
        "cached_queries": cached,
        "harvested_queries": harvested,
        "warning": "; ".join(warnings) if warnings else None,
        "at": iso_now(),
        "source": "xano",
        "mode": mode,
    }


def sync_jobs(query: str, location: str | None = None) -> dict[str, Any]:
    if not settings.live_harvest_enabled:
        return {"synced": 0, "query": query, "location": location, "source": "unconfigured"}
    return harvest_queries([SearchQuery(query, location)], mode="fanout", cache_hours=settings.harvest_cache_hours)


def fanout_jobs(
    target_title: str,
    location: str | None = None,
    top_skill: str | None = None,
    work_modes: list[str] | None = None,
) -> dict[str, Any]:
    title = (target_title or "").strip()
    if not title:
        raise ValueError("Set a target role so we can pull live jobs")
    if not settings.live_harvest_enabled:
        return {
            "synced": 0,
            "inserted": 0,
            "updated": 0,
            "embedded": 0,
            "cached_queries": 0,
            "harvested_queries": 0,
            "at": iso_now(),
            "source": "unconfigured",
        }
    queries = expand_queries(title, location, top_skill, work_modes)
    return harvest_queries(queries, mode="fanout", cache_hours=settings.harvest_cache_hours)


def expire_jobs() -> dict[str, int]:
    stale_cut = now_ms() - settings.harvest_stale_hours * 3600 * 1000
    expired_unseen = now_ms() - 7 * 24 * 3600 * 1000
    expired_posted = now_ms() - 30 * 24 * 3600 * 1000
    stale = expired = 0
    try:
        rows = xano.list_jobs()
    except XanoError:
        logger.exception("Expire pass could not list jobs")
        return {"stale": 0, "expired": 0}
    for row in rows:
        record_id = row.get("id")
        if record_id is None:
            continue
        status = str(row.get("status") or "active")
        last_seen = int(row.get("last_seen_at") or 0)
        posted = int(row.get("date_posted") or 0)
        next_status = status
        if not last_seen or last_seen < expired_unseen or (posted and posted < expired_posted):
            next_status = "expired"
        elif status == "active" and last_seen < stale_cut:
            next_status = "stale"
        if next_status == status:
            continue
        try:
            xano.patch_job(record_id, {"status": next_status})
        except XanoError:
            logger.exception("Could not mark job %s as %s", record_id, next_status)
            continue
        if next_status == "stale":
            stale += 1
        else:
            expired += 1
    return {"stale": stale, "expired": expired}


def purge_job_catalog() -> dict[str, int]:
    """Delete catalog rows and search-run cache so harvest can re-embed from scratch.

    If the jobs DELETE endpoint is not published, expire rows and bust content_hash
    so skip-on-duplicate-hash cannot keep empty embeddings forever.
    """
    deleted_jobs = expired_jobs = deleted_runs = 0
    can_delete = True
    try:
        jobs = xano.list_jobs()
    except XanoError:
        logger.exception("Purge could not list jobs")
        jobs = []
    for row in jobs:
        record_id = row.get("id")
        if record_id is None:
            continue
        if can_delete:
            try:
                xano.delete_job(record_id)
                deleted_jobs += 1
                continue
            except XanoError as exc:
                message = str(exc)
                if "Unable to locate request" in message or "→ 404" in message:
                    can_delete = False
                    logger.warning("Job DELETE endpoint is not published; expiring rows instead")
                else:
                    logger.exception("Could not delete job %s", record_id)
                    continue
        try:
            xano.patch_job(
                record_id,
                {
                    "status": "expired",
                    "content_hash": f"purged:{record_id}",
                    "apply_url": "",
                },
            )
            expired_jobs += 1
        except XanoError:
            logger.exception("Could not expire job %s", record_id)
    try:
        runs = xano.list_runs()
    except XanoError:
        logger.exception("Purge could not list search-runs")
        runs = []
    for row in runs:
        record_id = row.get("id")
        if record_id is None:
            continue
        try:
            xano.delete_run(record_id)
            deleted_runs += 1
        except XanoError:
            logger.exception("Could not delete search-run %s", record_id)
    _memory_runs.clear()
    logger.info(
        "Purged catalog: deleted_jobs=%s expired_jobs=%s deleted_runs=%s",
        deleted_jobs,
        expired_jobs,
        deleted_runs,
    )
    return {
        "deleted_jobs": deleted_jobs,
        "expired_jobs": expired_jobs,
        "deleted_runs": deleted_runs,
    }


def fetch_harvest_demand() -> dict[str, Any]:
    url = settings.backend_url.rstrip("/") + "/api/internal/harvest-demand"
    headers = {"X-Matchr-Service-Secret": settings.matchr_service_secret}
    try:
        response = httpx.get(url, headers=headers, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data
    except Exception:
        logger.exception("harvest-demand fetch failed")
    return {"titles": [], "locations": [], "user_count": 0}


def standing_harvest(demand: dict[str, Any] | None = None) -> dict[str, Any]:
    if not settings.live_harvest_enabled:
        return {"synced": 0, "source": "unconfigured", "at": iso_now()}
    snapshot = demand if demand is not None else fetch_harvest_demand()
    queries = allocate_standing_queries(snapshot, budget=settings.harvest_standing_slots)
    stats = harvest_queries(queries, mode="standing", cache_hours=settings.harvest_standing_hours)
    expiry = expire_jobs()
    stats["expire"] = expiry
    stats["slots"] = [{"q": query.q, "location": query.location} for query in queries]
    return stats
