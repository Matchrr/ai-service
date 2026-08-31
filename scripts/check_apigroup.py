"""Fetch API group details to find the correct base URL and branch."""
import os
import httpx
import json
from pathlib import Path

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


def _meta_url() -> str:
    api_url = os.environ.get("XANO_API_URL", "").rstrip("/")
    instance = api_url.split("https://")[-1].split("/")[0].replace(".xano.io", "")
    return f"https://{instance}.xano.io/api:meta"


def main() -> int:
    _load_dotenv()
    token = os.environ.get("XANO_METADATA_API_TOKEN", "").strip()
    base = _meta_url()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    print(f"Metadata API: {base}")
    for path in ["/workspace/1/apigroup/11", "/workspace/1/apigroup/11/api/55"]:
        try:
            r = httpx.get(f"{base}{path}", headers=headers, timeout=30.0)
            print(f"\nGET {path} -> {r.status_code}")
            print(json.dumps(r.json(), indent=2)[:2000])
        except Exception as e:
            print(f"\nGET {path} -> ERROR {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
