"""Create a test endpoint in v1 to verify the branch parameter works."""
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
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "text/x-xanoscript"}
    xs = '''query match_jobs_test verb=POST {
  description = "Test endpoint in v1."
  input { int x }
  stack { var $r { value = $input.x } }
  response = $r
}
'''
    for branch in ["v1", "v4"]:
        r = httpx.post(
            f"{base}/workspace/1/apigroup/11/api",
            headers=headers,
            content=xs,
            params={"branch": branch},
            timeout=30.0,
        )
        print(f"Create in {branch}: {r.status_code}")
        print(r.text[:1000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
