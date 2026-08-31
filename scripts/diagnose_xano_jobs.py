"""Diagnose why match_jobs returns empty.

This script queries the Xano matchrr_job_position table directly and reports:
- Total record count
- Count by status
- Most recent last_seen_at values
- Whether any records have embeddings
- Sample active records

Usage:
    XANO_API_KEY=your-key python scripts/diagnose_xano_jobs.py
"""

import os
import time
from collections import Counter
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
        "xano_api_url": os.environ.get("XANO_API_URL", "").rstrip("/"),
        "xano_api_key": os.environ.get("XANO_API_KEY", ""),
        "xano_jobs_table": os.environ.get("XANO_JOBS_TABLE", "matchrr_job_position"),
    }


def _get_table(
    url: str,
    api_key: str,
    table: str,
    per_page: int = 100,
    max_pages: int = 2,
) -> list[dict[str, Any]]:
    full_url = f"{url}/{table}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    records: list[dict[str, Any]] = []
    page = 1
    while page <= max_pages:
        response = httpx.get(
            full_url,
            headers=headers,
            params={"page": page, "per_page": per_page},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        batch = data if isinstance(data, list) else data.get("items") or data.get("records") or []
        if not batch:
            break
        records.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
    return records


def main() -> int:
    cfg = _settings()
    if not cfg["xano_api_url"]:
        print("ERROR: XANO_API_URL is not set. Check ai-service/.env.")
        return 1

    if cfg["xano_api_key"]:
        print("Auth: Authorization header will be used.")
    else:
        print("WARNING: XANO_API_KEY is not set. Calling without authentication.")
        print("         If the endpoint is private, set XANO_API_KEY and run again.")

    print(f"\nFetching records from {cfg['xano_api_url']}/{cfg['xano_jobs_table']} ...")
    try:
        records = _get_table(cfg["xano_api_url"], cfg["xano_api_key"], cfg["xano_jobs_table"])
    except Exception as exc:
        print(f"FAILED to fetch: {exc}")
        return 1

    print(f"\nRecords sampled: {len(records)}")
    if not records:
        print("The table is empty. Run a harvest or seed a test record.")
        return 0

    status_counts = Counter(str(r.get("status")) for r in records)
    print("\nStatus counts:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count}")

    last_seen_values = [int(r.get("last_seen_at") or 0) for r in records if r.get("last_seen_at")]
    if last_seen_values:
        print(f"\nlast_seen_at range:")
        print(f"  oldest: {min(last_seen_values)} ({_ms_to_date(min(last_seen_values))})")
        print(f"  newest: {max(last_seen_values)} ({_ms_to_date(max(last_seen_values))})")

    with_embedding = sum(1 for r in records if r.get("embedding") and isinstance(r.get("embedding"), list))
    print(f"\nRecords with embedding: {with_embedding}/{len(records)}")

    active_records = [r for r in records if str(r.get("status")) == "active"]
    print(f"\nRecords with status == 'active': {len(active_records)}")
    if active_records:
        print("Sample active record:")
        sample = active_records[0]
        print(f"  id: {sample.get('id')}")
        print(f"  external_id: {sample.get('external_id')}")
        print(f"  title: {sample.get('job_position_name')}")
        print(f"  company: {sample.get('company_name')}")
        print(f"  location: {sample.get('location')}")
        print(f"  work_mode: {sample.get('work_mode')}")
        print(f"  job_type: {sample.get('job_type')}")
        print(f"  last_seen_at: {sample.get('last_seen_at')} ({_ms_to_date(int(sample.get('last_seen_at') or 0))})")
        embedding = sample.get("embedding")
        if isinstance(embedding, list):
            print(f"  embedding length: {len(embedding)}")
            print(f"  embedding sample: {embedding[:5]}")

    now = int(time.time() * 1000)
    recent_active = [r for r in active_records if int(r.get("last_seen_at") or 0) > now - 7 * 24 * 3600 * 1000]
    print(f"\nActive records seen in last 7 days: {len(recent_active)}")

    return 0


def _ms_to_date(ms: int) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ms / 1000))
    except Exception:
        return "invalid"


if __name__ == "__main__":
    raise SystemExit(main())
