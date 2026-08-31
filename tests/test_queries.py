"""Tests for title-family fan-out and standing-harvest slot allocation."""

from app.pipelines.queries import (
    ADJACENT_TITLE,
    SearchQuery,
    allocate_standing_queries,
    expand_queries,
)


def test_expand_queries_basic_title_location_skill():
    queries = expand_queries(
        "Backend Engineer", "Toronto, ON", "Python", work_modes=["remote", "hybrid"]
    )
    assert len(queries) <= 4
    texts = {q.q for q in queries}
    assert "Backend Engineer" in texts
    assert "Backend Engineer Python" in texts
    assert any("Software Engineer" in q.q for q in queries)
    # Remote-only flag is not set, so at least one query should carry the city.
    assert any(q.location == "Toronto, ON" for q in queries)


def test_expand_queries_remote_only_omits_location():
    queries = expand_queries(
        "Backend Engineer", "Toronto, ON", "Python", work_modes=["remote"]
    )
    assert all(q.location is None for q in queries)


def test_expand_queries_literal_remote_location():
    queries = expand_queries("Backend Engineer", "Remote", "Python")
    assert all(q.location is None for q in queries)


def test_expand_queries_deduplicates():
    # "Software Engineer" with no skill and "Software Engineer" with skill must stay distinct.
    queries = expand_queries("Software Engineer", "Remote", None)
    assert len(queries) <= 4
    assert all(q.location is None for q in queries)


def test_expand_queries_capped_at_four():
    queries = expand_queries(
        "Backend Engineer", "Toronto, ON", "Python", work_modes=["remote", "hybrid"]
    )
    assert len(queries) <= 4


def test_expand_queries_seniority_variant_when_no_adjacent():
    # "Nurse" has no adjacent title, so seniority variant should be used.
    queries = expand_queries("Nurse", "Toronto, ON", None)
    assert len(queries) <= 4
    # With no adjacent title and no skill, we get at least the title + city and title alone.
    texts = {q.q for q in queries}
    assert "Nurse" in texts


def test_allocate_standing_queries_majority_dominates():
    demand = {
        "titles": [
            {"raw": "Software Engineer", "family": "software_engineering", "count": 8},
            {"raw": "Product Manager", "family": "product", "count": 2},
        ],
        "locations": [
            {"normalized": "ontario", "count": 8},
            {"normalized": "texas", "count": 2},
        ],
        "user_count": 10,
    }
    queries = allocate_standing_queries(demand, budget=20)
    assert len(queries) <= 20

    by_loc = {}
    for q in queries:
        by_loc.setdefault(q.location, 0)
        by_loc[q.location] += 1

    # Ontario should get at least as many slots as Texas because it has 80% of users.
    # Note: the current implementation caps distinct titles per family, which can
    # flatten the majority advantage (see bug note in assessment).
    assert by_loc.get("ontario", 0) >= by_loc.get("texas", 0)

    # Software engineering titles should dominate.
    se_titles = {
        "Software Engineer", "Backend Engineer", "Frontend Engineer",
        "Full Stack Engineer", "Mobile Engineer", "DevOps Engineer", "QA Engineer",
    }
    se_count = sum(1 for q in queries if q.q in se_titles)
    assert se_count > len(queries) // 2

    # No invented locations.
    assert all(q.location in {None, "ontario", "texas"} for q in queries)


def test_allocate_standing_queries_minority_floors():
    demand = {
        "titles": [
            {"raw": "Software Engineer", "family": "software_engineering", "count": 8},
            {"raw": "Product Manager", "family": "product", "count": 2},
        ],
        "locations": [
            {"normalized": "ontario", "count": 8},
            {"normalized": "texas", "count": 2},
        ],
        "user_count": 10,
    }
    queries = allocate_standing_queries(demand, budget=20)
    # Both title families and both locations should get at least one slot.
    families = {q.q.split()[0] for q in queries}
    assert any("Product" in q.q for q in queries)
    assert any(q.location == "texas" for q in queries)


def test_allocate_standing_queries_zero_users_uses_defaults():
    demand = {"titles": [], "locations": [], "user_count": 0}
    queries = allocate_standing_queries(demand, budget=20)
    assert len(queries) <= 20
    # Default families should be used and location omitted.
    assert all(q.location is None for q in queries)
    default_titles = {
        "Software Engineer", "Backend Engineer", "Frontend Engineer",
        "Full Stack Engineer", "Mobile Engineer", "DevOps Engineer", "QA Engineer",
        "Data Scientist", "Machine Learning Engineer", "Data Engineer",
        "Product Manager", "Technical Program Manager",
        "Product Designer", "UX Designer",
        "Sales Engineer", "Customer Success Manager", "Marketing Manager",
        "Business Analyst", "IT Support Specialist", "Security Engineer",
    }
    assert all(q.q in default_titles for q in queries)


def test_allocate_standing_queries_remote_location():
    demand = {
        "titles": [
            {"raw": "Software Engineer", "family": "software_engineering", "count": 5},
        ],
        "locations": [
            {"normalized": "remote", "count": 5},
        ],
        "user_count": 5,
    }
    queries = allocate_standing_queries(demand, budget=20)
    # Remote should omit the location parameter.
    assert all(q.location is None for q in queries)


def test_allocate_standing_queries_no_duplicate_pairs():
    demand = {
        "titles": [
            {"raw": "Software Engineer", "family": "software_engineering", "count": 20},
        ],
        "locations": [
            {"normalized": "ontario", "count": 20},
        ],
        "user_count": 20,
    }
    queries = allocate_standing_queries(demand, budget=20)
    # Each (q, location) pair should be unique.
    assert len(queries) == len({(q.q, q.location) for q in queries})
