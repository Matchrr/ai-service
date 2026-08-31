"""Create a temporary Xano endpoint to test a simple query without vector search."""
import os
import httpx
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AI_SERVICE = ROOT / "ai-service"

WORKSPACE_ID = 1
APIGROUP_ID = 12

FUNCTION_XS = """function match_jobs_simple_test {
  input {
    int cutoff
    int k?=50
  }
  stack {
    db.query matchrr_job_position {
      where = $db.matchrr_job_position.status == "active" && $db.matchrr_job_position.last_seen_at > $input.cutoff
      return = {
        type: "list"
        paging: {page: 1, per_page: $input.k, metadata: false}
      }
    } as $results
  }
  response = $results
}
"""

ENDPOINT_XS = """query match_jobs_simple_test verb=POST {
  input {
    int cutoff
    int k?=50
  }
  stack {
    function.run "match_jobs_simple_test" {
      input = {
        cutoff: $input.cutoff,
        k: $input.k
      }
    } as $matches
  }
  response = $matches
}
"""


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

    # Create or update function
    r = client.post(
        f"/workspace/{WORKSPACE_ID}/function",
        content=FUNCTION_XS,
        headers={"Content-Type": "text/x-xanoscript"},
    )
    print(f"Create function: {r.status_code}")
    if r.status_code >= 400:
        print(r.text)
        # Try to update existing function by finding it
        r2 = client.get(f"/workspace/{WORKSPACE_ID}/function", params={"per_page": 100})
        fn_id = None
        for f in r2.json().get("items", []):
            if f.get("name") == "match_jobs_simple_test":
                fn_id = f.get("id")
                r = client.put(
                    f"/workspace/{WORKSPACE_ID}/function/{fn_id}",
                    content=FUNCTION_XS,
                    headers={"Content-Type": "text/x-xanoscript"},
                    params={"publish": "true"},
                )
                print(f"Update function: {r.status_code}")
                if r.status_code >= 400:
                    print(r.text)
                break
    else:
        fn_id = r.json().get("id")

    # Create or update endpoint
    r = client.post(
        f"/workspace/{WORKSPACE_ID}/apigroup/{APIGROUP_ID}/api",
        content=ENDPOINT_XS,
        headers={"Content-Type": "text/x-xanoscript"},
        params={"branch": "v1"},
    )
    print(f"Create endpoint: {r.status_code}")
    if r.status_code >= 400:
        print(r.text)
        r2 = client.get(f"/workspace/{WORKSPACE_ID}/apigroup/{APIGROUP_ID}/api", params={"per_page": 100})
        ep_id = None
        for ep in r2.json().get("items", []):
            if ep.get("name") == "match_jobs_simple_test":
                ep_id = ep.get("id")
                r = client.put(
                    f"/workspace/{WORKSPACE_ID}/apigroup/{APIGROUP_ID}/api/{ep_id}",
                    content=ENDPOINT_XS,
                    headers={"Content-Type": "text/x-xanoscript"},
                    params={"publish": "true", "branch": "v1"},
                )
                print(f"Update endpoint: {r.status_code}")
                if r.status_code >= 400:
                    print(r.text)
                break
    else:
        ep_id = r.json().get("id")

    if ep_id:
        client.put(
            f"/workspace/{WORKSPACE_ID}/apigroup/{APIGROUP_ID}/api/{ep_id}",
            content=ENDPOINT_XS,
            headers={"Content-Type": "text/x-xanoscript"},
            params={"publish": "true", "branch": "v1"},
        )

    # Test via workspace API
    import time
    api_url = os.environ.get("XANO_API_URL", "").rstrip("/")
    key = os.environ.get("XANO_API_KEY", "")
    headers2 = {"Content-Type": "application/json"}
    if key:
        headers2["Authorization"] = f"Bearer {key}"
    cutoff = int(time.time() * 1000) - 7 * 24 * 3600 * 1000
    r = httpx.post(
        f"{api_url}/match_jobs_simple_test",
        headers=headers2,
        json={"cutoff": cutoff, "k": 10},
        timeout=30.0,
    )
    print(f"\nTest endpoint: {r.status_code}")
    print(r.text[:1000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
