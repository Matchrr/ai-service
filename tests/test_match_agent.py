"""Tests for the SemanticMatchAgent retrieval + semantic scoring."""

import time
from unittest.mock import MagicMock

import pytest

from app.agents import match as match_module
from app.agents.match import SemanticMatchAgent


@pytest.fixture
def xano_mock(monkeypatch):
    mock = MagicMock()
    mock.vector_search_jobs.return_value = None  # default to brute-force path
    monkeypatch.setattr(match_module, "xano", mock)
    return mock


@pytest.fixture
def embed_mock(monkeypatch):
    monkeypatch.setattr(match_module, "embed_text", lambda text, task_type=None: [1.0] + [0.0] * 1535)


@pytest.fixture
def settings_mock(ai_settings):
    return ai_settings(harvest_stale_hours=96)


def _job_row(
    job_id,
    status="active",
    last_seen_hours=2,
    work_mode="remote",
    job_type="full_time",
    pay_min=100_000,
    pay_max=None,
    title="Software Engineer",
    required_skills=None,
    location="Remote",
    embedding=None,
):
    now = int(time.time() * 1000)
    return {
        "id": job_id,
        "external_id": f"ext-{job_id}",
        "job_position_name": title,
        "company_name": "Acme",
        "location": location,
        "job_description": "Build systems.",
        "status": status,
        "last_seen_at": now - last_seen_hours * 3600 * 1000,
        "work_mode": work_mode,
        "job_type": job_type,
        "pay_min": pay_min,
        "pay_max": pay_max if pay_max is not None else pay_min + 20_000,
        "pay_type": "annual",
        "required_skills": required_skills or [],
        "embedding": embedding or [1.0] + [0.0] * 1535,
    }


def test_match_excludes_stale_jobs(xano_mock, embed_mock, settings_mock):
    xano_mock.list_jobs.return_value = [
        _job_row(1, last_seen_hours=2),   # active and recent
        _job_row(2, last_seen_hours=200), # stale (>96h)
    ]
    result = SemanticMatchAgent().run("cand-1", 10, profile_text="software engineer python")
    ids = {m["id"] for m in result["matches"]}
    assert "ext-1" in ids
    assert "ext-2" not in ids
    assert result["pool_size"] == 1


def test_match_excludes_non_active_status(xano_mock, embed_mock, settings_mock):
    xano_mock.list_jobs.return_value = [
        _job_row(1, status="active"),
        _job_row(2, status="expired"),
        _job_row(3, status="stale"),
    ]
    result = SemanticMatchAgent().run("cand-1", 10, profile_text="software engineer")
    assert result["pool_size"] == 1


def test_match_excludes_missing_external_id(xano_mock, embed_mock, settings_mock):
    row = _job_row(1)
    row["external_id"] = ""
    xano_mock.list_jobs.return_value = [row]
    result = SemanticMatchAgent().run("cand-1", 10, profile_text="software engineer")
    assert result["pool_size"] == 0


def test_match_work_mode_filter(xano_mock, embed_mock, settings_mock):
    xano_mock.list_jobs.return_value = [
        _job_row(1, work_mode="remote"),
        _job_row(2, work_mode="onsite"),
    ]
    result = SemanticMatchAgent().run(
        "cand-1", 10, profile_text="software engineer", work_modes=["remote"]
    )
    assert result["pool_size"] == 1
    assert result["matches"][0]["work_mode"] == "remote"


def test_match_job_type_filter(xano_mock, embed_mock, settings_mock):
    xano_mock.list_jobs.return_value = [
        _job_row(1, job_type="full_time"),
        _job_row(2, job_type="contract"),
    ]
    result = SemanticMatchAgent().run(
        "cand-1", 10, profile_text="software engineer", job_types=["contract"]
    )
    assert result["pool_size"] == 1
    assert result["matches"][0]["job_type"] == "contract"


def test_match_pay_min_filter_uses_top_of_range(xano_mock, embed_mock, settings_mock):
    xano_mock.list_jobs.return_value = [
        _job_row(1, pay_min=80_000, pay_max=130_000),
        _job_row(2, pay_min=40_000, pay_max=90_000),
    ]
    result = SemanticMatchAgent().run(
        "cand-1", 10, profile_text="software engineer", pay_min=100_000, pay_period="annual"
    )
    assert result["pool_size"] == 1
    assert result["matches"][0]["pay_min"] == 80_000


def test_match_pay_min_filter_excludes_below_floor(xano_mock, embed_mock, settings_mock):
    xano_mock.list_jobs.return_value = [
        _job_row(1, pay_min=50_000, pay_max=90_000),
        _job_row(2, pay_min=120_000, pay_max=150_000),
    ]
    result = SemanticMatchAgent().run(
        "cand-1", 10, profile_text="software engineer", pay_min=100_000, pay_period="annual"
    )
    assert result["pool_size"] == 1
    assert result["matches"][0]["pay_min"] == 120_000


def test_match_empty_profile_returns_unranked(xano_mock, settings_mock):
    xano_mock.list_jobs.return_value = [_job_row(1)]
    result = SemanticMatchAgent().run("cand-1", 10, profile_text="")
    assert result["pool_size"] == 1
    assert result["matches"][0]["similarity"] == 0.0


def test_match_returns_external_id_as_id(xano_mock, embed_mock, settings_mock):
    xano_mock.list_jobs.return_value = [_job_row(1)]
    result = SemanticMatchAgent().run("cand-1", 10, profile_text="software engineer")
    assert result["matches"][0]["id"] == "ext-1"
    assert result["matches"][0]["external_id"] == "ext-1"


def test_match_fallback_on_xano_error(xano_mock, settings_mock):
    xano_mock.vector_search_jobs.return_value = None
    xano_mock.list_jobs.side_effect = match_module.XanoError("xano down")
    result = SemanticMatchAgent().run("cand-1", 10, profile_text="software engineer")
    assert result["matches"] == []
    assert result["source"] == "error"


def test_match_uses_xano_vector_search_when_configured(xano_mock, embed_mock, ai_settings):
    ai_settings(xano_match_endpoint="match_jobs", harvest_stale_hours=96)
    xano_mock.vector_search_jobs.return_value = [
        _job_row(1, title="Software Engineer"),
        _job_row(2, title="Data Scientist"),
    ]
    xano_mock.list_jobs.return_value = []  # should not be used

    result = SemanticMatchAgent().run(
        "cand-1", 10, profile_text="software engineer python", target_title="Software Engineer"
    )
    assert result["source"] == "xano_vector"
    assert result["pool_size"] == 2
    xano_mock.list_jobs.assert_not_called()


def test_match_vector_search_results_are_filtered(xano_mock, embed_mock, ai_settings):
    ai_settings(xano_match_endpoint="match_jobs", harvest_stale_hours=96)
    xano_mock.vector_search_jobs.return_value = [
        _job_row(1, last_seen_hours=2),   # recent
        _job_row(2, last_seen_hours=200), # stale
    ]

    result = SemanticMatchAgent().run("cand-1", 10, profile_text="software engineer")
    assert result["pool_size"] == 1
    assert result["matches"][0]["id"] == "ext-1"


def test_match_semantic_score_prefers_title_and_skill_overlap(xano_mock, embed_mock, settings_mock):
    # Same embedding for all rows; title and skill overlap should determine the ranking.
    xano_mock.list_jobs.return_value = [
        _job_row(1, title="Software Engineer", required_skills=["Python", "AWS"]),
        _job_row(2, title="Marketing Manager", required_skills=["SEO", "Content"]),
        _job_row(3, title="Software Engineer", required_skills=["Java", "Spring"]),
    ]
    result = SemanticMatchAgent().run(
        "cand-1",
        10,
        profile_text="software engineer",
        target_title="Software Engineer",
        skills=["Python", "AWS"],
    )
    assert result["matches"][0]["title"] == "Software Engineer"
    assert result["matches"][0]["semantic_score"] > result["matches"][1]["semantic_score"]
    assert all("semantic_score" in m for m in result["matches"])


def test_match_location_filter_boosts_preferred_location(xano_mock, embed_mock, settings_mock):
    xano_mock.list_jobs.return_value = [
        _job_row(1, location="Toronto, ON"),
        _job_row(2, location="Remote"),
    ]
    result = SemanticMatchAgent().run(
        "cand-1", 10, profile_text="software engineer", location="Toronto, ON"
    )
    # The Toronto job should rank higher because of location match.
    assert result["matches"][0]["location"] == "Toronto, ON"


def test_match_vector_search_called_with_filters(ai_settings, xano_mock, embed_mock):
    ai_settings(xano_match_endpoint="match_jobs", harvest_stale_hours=96, xano_api_url="https://test.xano.io/api:test")
    xano_mock.vector_search_jobs.return_value = []

    SemanticMatchAgent().run(
        "cand-1",
        10,
        profile_text="software engineer",
        work_modes=["remote"],
        job_types=["full_time"],
        pay_min=100_000,
        pay_period="annual",
    )
    assert xano_mock.vector_search_jobs.called
    call_kwargs = xano_mock.vector_search_jobs.call_args[1]
    assert call_kwargs["k"] == 50
    assert call_kwargs["filters"]["status"] == "active"
    assert "last_seen_at" in call_kwargs["filters"]
    assert call_kwargs["filters"]["work_mode"]["in"] == ["remote"]
    assert call_kwargs["filters"]["job_type"]["in"] == ["full_time"]
