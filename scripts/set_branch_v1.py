"""Set v1 as the live branch for the workspace."""
import os
import httpx
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
    r = httpx.post(f"{base}/workspace/1/branch/v1/live", headers=headers, timeout=30.0)
    print(f"Set v1 live: {r.status_code}")
    print(r.text[:1000])
    return 0 if r.status_code < 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())
