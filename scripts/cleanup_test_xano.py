"""Remove temporary Xano test function and endpoint."""
import os
import httpx
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AI_SERVICE = ROOT / "ai-service"
WORKSPACE_ID = 1
APIGROUP_ID = 12


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


def _meta_url() -> str:
    api_url = os.environ.get("XANO_API_URL", "").rstrip("/")
    instance = api_url.split("https://")[-1].split("/")[0].replace(".xano.io", "")
    return f"https://{instance}.xano.io/api:meta"


def main() -> int:
    _load_dotenv()
    token = os.environ.get("XANO_METADATA_API_TOKEN", "").strip()
    base = _meta_url()
    headers = {"Authorization": f"Bearer {token}"}
    client = httpx.Client(base_url=base, headers=headers, timeout=60.0)

    # Delete function
    r = client.get(f"/workspace/{WORKSPACE_ID}/function", params={"per_page": 100})
    for f in r.json().get("items", []):
        if f.get("name") == "match_jobs_simple_test":
            dr = client.delete(f"/workspace/{WORKSPACE_ID}/function/{f['id']}")
            print(f"Deleted function {f['id']}: {dr.status_code}")

    # Delete endpoint
    r = client.get(f"/workspace/{WORKSPACE_ID}/apigroup/{APIGROUP_ID}/api", params={"per_page": 100})
    for ep in r.json().get("items", []):
        if ep.get("name") == "match_jobs_simple_test":
            dr = client.delete(f"/workspace/{WORKSPACE_ID}/apigroup/{APIGROUP_ID}/api/{ep['id']}")
            print(f"Deleted endpoint {ep['id']}: {dr.status_code}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
