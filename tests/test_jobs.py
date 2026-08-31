"""Tests for job normalization, hashing, upsert rules, and expiry."""

import time
from unittest.mock import MagicMock

import pytest

from app.pipelines import jobs as jobs_module
from app.pipelines.jobs import (
    content_hash,
    expire_jobs,
    fingerprint_job,
    normalize_job,
    parse_relative_posted_at,
    parse_salary,
    upsert_jobs,
)
from app.pipelines.queries import SearchQuery


def test_fingerprint_job_uses_city_only():
    assert fingerprint_job("Acme Inc", "Software Engineer", "Toronto, ON, Canada") == \
        "acme-inc|software-engineer|toronto"


def test_content_hash_is_stable_and_case_insensitive():
    a = content_hash("  Build \n scalable   systems  ")
    b = content_hash("build scalable systems")
    assert a == b
    assert len(a) == 64


def test_parse_relative_posted_at_today():
    now = int(time.time() * 1000)
    parsed = parse_relative_posted_at("just posted")
    assert parsed is not None
    assert abs(parsed - now) < 60_000


def test_parse_relative_posted_at_days_ago():
    parsed = parse_relative_posted_at("3 days ago")
    assert parsed is not None
    expected = int(time.time() * 1000) - 3 * 24 * 3600 * 1000
    assert abs(parsed - expected) < 60_000


def test_parse_salary_hourly_range():
    assert parse_salary("$50 - $70 / hr") == (50.0, 70.0, "hourly")


def test_parse_salary_annual_range():
    assert parse_salary("$100k - $150k a year") == (100_000.0, 150_000.0, "annual")


def test_parse_salary_k_suffix():
    assert parse_salary("120k") == (120_000.0, 120_000.0, "annual")


def test_normalize_job_maps_fields():
    raw = {
        "title": "Software Engineer",
        "company_name": "Acme",
        "location": "Toronto, ON",
        "description": "Build systems.",
        "via": "Indeed",
        "job_id": "abc123",
        "detected_extensions": {
            "posted_at": "2 days ago",
            "schedule_type": "Full-time",
            "salary": "$100k - $130k a year",
            "qualifications": "Mid-level",
        },
        "job_highlights": [
            {"title": "Qualifications", "items": ["Python", "AWS"]},
            {"title": "Benefits", "items": ["Health insurance"]},
        ],
        "apply_options": [
            {"title": "Apply on company site", "link": "https://acme.com/jobs/123"},
            {"title": "Apply on Indeed", "link": "https://indeed.com/apply"},
        ],
    }
    job = normalize_job(raw)
    assert job["job_position_name"] == "Software Engineer"
    assert job["company_name"] == "Acme"
    assert job["external_id"] == "abc123"
    assert job["job_source"] == "Indeed"
    assert job["job_type"] == "full_time"
    assert job["pay_type"] == "annual"
    assert job["work_mode"] == "onsite"
    assert "Health insurance" in job["benefits"]
    assert "Python" in job["required_skills"]
    assert job["apply_url"].startswith("https://acme.com")


@pytest.fixture
def xano_mock(monkeypatch):
    mock = MagicMock()
    mock.list_jobs.return_value = []
    mock.add_job.side_effect = lambda payload: {"id": 1, **payload}
    mock.patch_job.side_effect = lambda record_id, payload: {"id": record_id, **payload}
    monkeypatch.setattr(jobs_module, "xano", mock)
    return mock


@pytest.fixture
def embed_mock(monkeypatch):
    def _embed(texts, *, task_type=None):
        return [[0.0] * 1536 for _ in texts]
    monkeypatch.setattr(jobs_module, "embed_texts", _embed)


@pytest.fixture
def settings_for_upsert(ai_settings):
    return ai_settings(
        harvest_stale_hours=96,
        embedding_dimensions=1536,
    )


def _raw_job(
    job_id="ext1",
    title="Software Engineer",
    company="Acme",
    location="Toronto, ON",
    description="Build systems.",
    apply_url="https://acme.com/apply",
):
    return {
        "title": title,
        "company_name": company,
        "location": location,
        "description": description,
        "via": "Indeed",
        "job_id": job_id,
        "detected_extensions": {"posted_at": "1 day ago", "schedule_type": "Full-time"},
        "apply_options": [{"title": "Apply", "link": apply_url}],
    }


def test_upsert_new_job_inserts_and_embeds(xano_mock, embed_mock, settings_for_upsert):
    stats = upsert_jobs([_raw_job()])
    assert stats["inserted"] == 1
    assert stats["updated"] == 0
    assert stats["embedded"] == 1
    assert xano_mock.add_job.called


def test_upsert_same_external_id_same_hash_skips_embed(xano_mock, embed_mock, settings_for_upsert):
    existing = normalize_job(_raw_job())
    existing["id"] = 1
    existing["embedding"] = [0.0] * 1536
    xano_mock.list_jobs.return_value = [existing]

    stats = upsert_jobs([_raw_job()])
    assert stats["inserted"] == 0
    assert stats["updated"] == 1
    assert stats["embedded"] == 0


def test_upsert_same_external_id_changed_hash_reembeds(xano_mock, embed_mock, settings_for_upsert):
    existing = normalize_job(_raw_job(description="Old description"))
    existing["id"] = 1
    existing["embedding"] = [0.0] * 1536
    xano_mock.list_jobs.return_value = [existing]

    stats = upsert_jobs([_raw_job(description="Brand new description")])
    assert stats["inserted"] == 0
    assert stats["updated"] == 1
    assert stats["embedded"] == 1


def test_upsert_same_fingerprint_different_external_id_merges(xano_mock, embed_mock, settings_for_upsert):
    existing = normalize_job(_raw_job(job_id="ext1", apply_url="https://acme.com/old"))
    existing["id"] = 1
    existing["embedding"] = [0.0] * 1536
    xano_mock.list_jobs.return_value = [existing]

    stats = upsert_jobs([_raw_job(job_id="ext2", apply_url="https://acme.com/new")])
    assert stats["inserted"] == 0
    assert stats["updated"] == 1
    assert stats["embedded"] == 0


def test_upsert_missing_from_scrape_left_in_place(xano_mock, embed_mock, settings_for_upsert):
    existing = normalize_job(_raw_job(job_id="ext1"))
    existing["id"] = 1
    existing["status"] = "active"
    existing["last_seen_at"] = 1  # very old
    xano_mock.list_jobs.return_value = [existing]

    stats = upsert_jobs([])
    # Missing from scrape should not delete or update the row.
    assert stats["inserted"] == 0
    assert stats["updated"] == 0
    assert xano_mock.patch_job.call_count == 0


def test_upsert_junk_apply_url_is_dropped(xano_mock, embed_mock, settings_for_upsert):
    raw = _raw_job()
    raw["apply_options"] = [{"title": "Apply", "link": "https://bebee.com/apply"}]
    stats = upsert_jobs([raw])
    assert stats["inserted"] == 0
    assert stats["skipped_junk"] == 1


def test_expire_marks_stale_after_96h(ai_settings, xano_mock):
    ai_settings(harvest_stale_hours=96)
    now = int(time.time() * 1000)
    stale = now - 100 * 3600 * 1000
    row = {"id": 1, "status": "active", "last_seen_at": stale, "date_posted": now}
    xano_mock.list_jobs.return_value = [row]

    stats = expire_jobs()
    assert stats["stale"] == 1
    assert stats["expired"] == 0
    xano_mock.patch_job.assert_called_once()
    assert xano_mock.patch_job.call_args[0][1]["status"] == "stale"


def test_expire_marks_expired_after_7d(ai_settings, xano_mock):
    ai_settings(harvest_stale_hours=96)
    now = int(time.time() * 1000)
    old = now - 8 * 24 * 3600 * 1000
    row = {"id": 1, "status": "active", "last_seen_at": old, "date_posted": now}
    xano_mock.list_jobs.return_value = [row]

    stats = expire_jobs()
    assert stats["stale"] == 0
    assert stats["expired"] == 1
    xano_mock.patch_job.assert_called_once()
    assert xano_mock.patch_job.call_args[0][1]["status"] == "expired"


def test_expire_marks_expired_for_ancient_posting(ai_settings, xano_mock):
    ai_settings(harvest_stale_hours=96)
    now = int(time.time() * 1000)
    recent_seen = now - 2 * 24 * 3600 * 1000
    old_posted = now - 31 * 24 * 3600 * 1000
    row = {"id": 1, "status": "active", "last_seen_at": recent_seen, "date_posted": old_posted}
    xano_mock.list_jobs.return_value = [row]

    stats = expire_jobs()
    assert stats["expired"] == 1


def test_cache_hit_skips_serpapi(xano_mock, ai_settings, monkeypatch):
    ai_settings(harvest_cache_hours=12, harvest_stale_hours=96)
    now = int(time.time() * 1000)
    xano_mock.list_runs.return_value = [
        {
            "id": 1,
            "q": "Software Engineer",
            "location": "Toronto, ON",
            "chips": "date_posted:week",
            "error": "",
            "cached_until": now + 10 * 3600 * 1000,
        }
    ]
    xano_mock.list_jobs.return_value = [{"id": 1, "external_id": "x"}]

    fetch_mock = MagicMock(return_value=([], "", None))
    monkeypatch.setattr(jobs_module, "fetch_google_jobs", fetch_mock)

    result = jobs_module.harvest_queries(
        [SearchQuery("Software Engineer", "Toronto, ON")], mode="fanout", cache_hours=12
    )
    assert result["cached_queries"] == 1
    assert result["harvested_queries"] == 0
    fetch_mock.assert_not_called()


def test_harvest_cold_catalog_ignores_cache(xano_mock, ai_settings, monkeypatch):
    ai_settings(harvest_cache_hours=12, harvest_stale_hours=96)
    xano_mock.list_jobs.return_value = []  # empty catalog
    xano_mock.list_runs.return_value = [
        {
            "id": 1,
            "q": "Software Engineer",
            "location": "Toronto, ON",
            "chips": "date_posted:week",
            "error": "",
            "cached_until": int(time.time() * 1000) + 10 * 3600 * 1000,
        }
    ]

    fetch_mock = MagicMock(return_value=([], "", None))
    monkeypatch.setattr(jobs_module, "fetch_google_jobs", fetch_mock)

    result = jobs_module.harvest_queries(
        [SearchQuery("Software Engineer", "Toronto, ON")], mode="fanout", cache_hours=12
    )
    # Catalog is empty, so cache is ignored to bootstrap.
    assert result["harvested_queries"] == 1
    fetch_mock.assert_called_once()
