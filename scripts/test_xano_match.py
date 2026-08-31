"""Automated test for the Xano match_jobs vector search function.

Usage:
    XANO_API_KEY=your-key python scripts/test_xano_match.py

The script reads XANO_API_URL and XANO_MATCH_ENDPOINT from ai-service/.env.
It can optionally seed a test job record if the table appears empty.
"""

import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Allow running from the repo root or from ai-service/scripts.
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
        "xano_api_url": os.environ.get("XANO_API_URL", "").rstrip("/"),
        "xano_api_key": os.environ.get("XANO_API_KEY", ""),
        "xano_match_endpoint": os.environ.get("XANO_MATCH_ENDPOINT", "match_jobs"),
    }


def _random_embedding(dim: int = 1536) -> list[float]:
    return [round(random.uniform(-0.1, 0.1), 4) for _ in range(dim)]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _call_xano(
    url: str,
    api_key: str,
    function: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    full_url = f"{url}/{function.lstrip('/')}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = httpx.post(full_url, headers=headers, json=payload, timeout=30.0)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"HTTP {exc.response.status_code}: {exc.response.text[:500]}")
        if exc.response.status_code == 404:
            print(
                "\nTIP: A 404 means the endpoint does not exist or is not published."
                "\n     In Xano, create an API endpoint (e.g., POST /match_jobs) that runs"
                "\n     the custom function, then publish it. Set XANO_MATCH_ENDPOINT to its path."
            )
        raise
    return response.json()


def _seed_test_record(url: str, api_key: str, table: str = "matchrr_job_position") -> dict[str, Any]:
    """Insert one test job record so match_jobs has something to find."""
    payload = {
        "job_position_name": "Software Engineer",
        "company_name": "Matchr Test Co",
        "location": "Remote",
        "job_description": "Build scalable systems with Python and AWS.",
        "job_source": "test",
        "external_id": f"test-{int(time.time())}",
        "status": "active",
        "last_seen_at": _now_ms(),
        "work_mode": "remote",
        "job_type": "full_time",
        "pay_min": 100000,
        "pay_max": 150000,
        "pay_type": "annual",
        "embedding": _random_embedding(),
    }
    full_url = f"{url}/{table}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = httpx.post(full_url, headers=headers, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def main() -> int:
    cfg = _settings()
    if not cfg["xano_api_url"]:
        print("ERROR: XANO_API_URL is not set. Check ai-service/.env.")
        return 1

    print(f"Xano API URL: {cfg['xano_api_url']}")
    print(f"Endpoint: {cfg['xano_match_endpoint']}")
    if cfg["xano_api_key"]:
        print("Auth: Authorization header will be used.")
    else:
        print("WARNING: XANO_API_KEY is not set. Calling without authentication.")
        print("         If the endpoint is private, set XANO_API_KEY and run again.")

    # Generate a test candidate embedding.
    embedding = _random_embedding()
    cutoff = _now_ms() - 7 * 24 * 3600 * 1000  # one week ago

    payload = {
        "embedding": embedding,
        "filters": {
            "last_seen_at": {"gt": cutoff},
        },
        "k": 10,
    }

    print(f"\nCalling POST /{cfg['xano_match_endpoint']} with last_seen_at.gt={cutoff} ...")
    try:
        result = _call_xano(
            cfg["xano_api_url"],
            cfg["xano_api_key"],
            cfg["xano_match_endpoint"],
            payload,
        )
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        return 1

    # Xano may return a list directly or wrap it in items/records.
    matches = result
    if isinstance(result, dict):
        for key in ("items", "records", "result", "payload"):
            if isinstance(result.get(key), list):
                matches = result[key]
                break

    print(f"\nResult type: {type(result).__name__}")
    if isinstance(matches, list):
        print(f"Matches returned: {len(matches)}")
        for match in matches[:3]:
            title = match.get("job_position_name") or match.get("title")
            company = match.get("company_name") or match.get("company")
            distance = match.get("distance")
            external_id = match.get("external_id")
            print(f"  - {title} at {company} (external_id={external_id}, distance={distance})")
    else:
        print(f"Unexpected result: {result}")
        return 1

    if len(matches) == 0 and "--seed" in sys.argv:
        print("\nNo matches found. Seeding a test record...")
        record = _seed_test_record(cfg["xano_api_url"], cfg["xano_api_key"])
        print(f"Seeded record id={record.get('id')}, external_id={record.get('external_id')}")
        print("Run the script again without --seed to test against the new record.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
