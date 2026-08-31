"""Automated test for the ai-service /api/agents/match endpoint.

This script calls the local ai-service match agent and reports whether it
uses Xano vector search or brute-force fallback, and how many jobs it returns.

Usage:
    AI_SERVICE_URL=http://localhost:8080 python scripts/test_ai_match.py

Make sure the Backend and ai-service are running, and the candidate has a
`target_title` set in the Backend store (or use a real candidate profile).
"""

import os
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
AI_SERVICE = ROOT / "ai-service"


def _load_dotenv() -> None:
    env_path = AI_SERVICE / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and value and key not in os.environ:
            os.environ[key] = value


def _settings() -> dict[str, str]:
    _load_dotenv()
    return {
        "ai_service_url": os.environ.get("AI_SERVICE_URL", "http://localhost:8080").rstrip("/"),
    }


def _call_agent(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    full_url = f"{url}/api/agents/match"
    response = httpx.post(
        full_url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60.0,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"HTTP {exc.response.status_code}: {exc.response.text[:500]}")
        raise
    return response.json()


def main() -> int:
    cfg = _settings()
    print(f"AI service URL: {cfg['ai_service_url']}")

    # Synthetic candidate profile for testing.
    payload = {
        "candidate_id": "test-candidate",
        "target_title": "Software Engineer",
        "profile_text": (
            "Software engineer with Python, AWS, and FastAPI experience. "
            "Built scalable backend services and REST APIs."
        ),
        "skills": ["Python", "AWS", "FastAPI", "PostgreSQL", "Docker"],
        "work_modes": ["remote", "hybrid"],
        "job_types": ["full-time"],
        "limit": 10,
    }

    print("\nCalling /api/agents/match ...")
    try:
        result = _call_agent(cfg["ai_service_url"], payload)
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        return 1

    print(f"\nAgent: {result.get('agent')}")
    print(f"Source: {result.get('source')}")
    print(f"Pool size: {result.get('pool_size')}")
    print(f"Limit: {result.get('limit')}")

    matches = result.get("matches") or []
    print(f"Matches returned: {len(matches)}")

    for match in matches[:5]:
        title = match.get("title")
        company = match.get("company")
        similarity = match.get("similarity")
        semantic = match.get("semantic_score")
        external_id = match.get("external_id")
        print(
            f"  - {title} at {company} "
            f"(external_id={external_id}, similarity={similarity}, semantic_score={semantic})"
        )

    if result.get("source") == "xano_vector":
        print("\n✅ Xano vector search is being used.")
    elif result.get("source") == "brute_force":
        print("\n⚠️  Falling back to brute-force cosine. Check XANO_MATCH_ENDPOINT and the Xano match endpoint.")
    else:
        print(f"\n⚠️  Unexpected source: {result.get('source')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
