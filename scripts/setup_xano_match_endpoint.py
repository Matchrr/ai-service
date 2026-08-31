"""Create the Xano match_jobs custom function and its API endpoint.

Uses the Metadata API token in ai-service/.env.
"""
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
AI_SERVICE = ROOT / "ai-service"

WORKSPACE_ID = 1
APIGROUP_ID = 12  # Matchr (v1 live branch)

FUNCTION_XS = """function match_jobs {
  description = "Vector search over active job positions using cosine similarity."
  input {
    decimal[] embedding
    json filters?
    int k?=50
  }
  stack {
    var $cutoff {
      value = $input.filters | get:"last_seen_at":{} | get:"gt":0
    }

    db.query matchrr_job_position {
      where = $db.matchrr_job_position.status == "active" && $db.matchrr_job_position.last_seen_at > $cutoff
      sort = {distance: "asc"}
      eval = {
        distance: $db.matchrr_job_position.embedding|cosine_distance:$input.embedding
      }
      return = {
        type: "list"
        paging: {page: 1, per_page: $input.k, metadata: false}
      }
    } as $results
  }
  response = $results
}
"""

ENDPOINT_XS = """query match_jobs verb=POST {
  description = "Vector search for job matching."
  input {
    decimal[] embedding
    json filters?
    int k?=50
  }
  stack {
    function.run "match_jobs" {
      input = {
        embedding: $input.embedding,
        filters: $input.filters,
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


def _client() -> tuple[httpx.Client, str]:
    _load_dotenv()
    token = os.environ.get("XANO_METADATA_API_TOKEN", "").strip()
    if not token:
        print("ERROR: XANO_METADATA_API_TOKEN is not set.")
        sys.exit(1)
    base = _meta_url()
    headers = {"Authorization": f"Bearer {token}"}
    return httpx.Client(base_url=base, headers=headers, timeout=60.0), base


def _unwrap_list(resp: httpx.Response, key: str = "items") -> list[dict]:
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return data.get(key, [])


def find_function(client: httpx.Client, name: str) -> int | None:
    page = 1
    while True:
        r = client.get(f"/workspace/{WORKSPACE_ID}/function", params={"page": page, "per_page": 100})
        items = _unwrap_list(r)
        for f in items:
            if f.get("name") == name:
                return f.get("id")
        if len(items) < 100:
            break
        page += 1
    return None


def find_endpoints(client: httpx.Client, path: str) -> list[int]:
    r = client.get(f"/workspace/{WORKSPACE_ID}/apigroup/{APIGROUP_ID}/api", params={"per_page": 100})
    items = _unwrap_list(r)
    return [ep.get("id") for ep in items if ep.get("name") == path]


def delete_endpoint(client: httpx.Client, api_id: int) -> None:
    r = client.delete(f"/workspace/{WORKSPACE_ID}/apigroup/{APIGROUP_ID}/api/{api_id}")
    print(f"Delete endpoint {api_id}: {r.status_code}")
    if r.status_code >= 400:
        print(r.text)
        r.raise_for_status()


def create_function(client: httpx.Client) -> int:
    r = client.post(
        f"/workspace/{WORKSPACE_ID}/function",
        content=FUNCTION_XS,
        headers={"Content-Type": "text/x-xanoscript"},
    )
    print(f"Create function response: {r.status_code}")
    if r.status_code >= 400:
        print(r.text)
        r.raise_for_status()
    data = r.json()
    print(f"Function created: id={data.get('id')} name={data.get('name')}")
    return data.get("id")


def update_function(client: httpx.Client, function_id: int) -> int:
    r = client.put(
        f"/workspace/{WORKSPACE_ID}/function/{function_id}",
        content=FUNCTION_XS,
        headers={"Content-Type": "text/x-xanoscript"},
        params={"publish": "true"},
    )
    print(f"Update function response: {r.status_code}")
    if r.status_code >= 400:
        print(r.text)
        r.raise_for_status()
    data = r.json()
    print(f"Function updated: id={data.get('id')} name={data.get('name')}")
    return data.get("id")


def create_endpoint(client: httpx.Client, branch: str = "v1") -> int:
    r = client.post(
        f"/workspace/{WORKSPACE_ID}/apigroup/{APIGROUP_ID}/api",
        content=ENDPOINT_XS,
        headers={"Content-Type": "text/x-xanoscript"},
        params={"branch": branch},
    )
    print(f"Create endpoint (branch={branch}) response: {r.status_code}")
    if r.status_code >= 400:
        print(r.text)
        r.raise_for_status()
    data = r.json()
    print(f"Endpoint created: id={data.get('id')} name={data.get('name')}")
    return data.get("id")


def update_endpoint(client: httpx.Client, api_id: int, branch: str = "v1") -> None:
    r = client.put(
        f"/workspace/{WORKSPACE_ID}/apigroup/{APIGROUP_ID}/api/{api_id}",
        content=ENDPOINT_XS,
        headers={"Content-Type": "text/x-xanoscript"},
        params={"publish": "true", "branch": branch},
    )
    print(f"Publish endpoint (branch={branch}) response: {r.status_code}")
    if r.status_code >= 400:
        print(r.text)
        r.raise_for_status()


def main() -> int:
    client, base = _client()
    print(f"Metadata API: {base}")

    existing_fn = find_function(client, "match_jobs")
    if existing_fn:
        print(f"match_jobs function already exists (id={existing_fn}); updating.")
        update_function(client, existing_fn)
    else:
        print("Creating match_jobs function...")
        create_function(client)

    # Ensure endpoint exists in the v1 branch (the branch the app calls).
    # Remove any stale match_jobs endpoints so we can recreate in v1.
    stale_ids = find_endpoints(client, "match_jobs") + find_endpoints(client, "match_jobs_test")
    for api_id in stale_ids:
        delete_endpoint(client, api_id)

    print("Creating POST /match_jobs endpoint in v1...")
    api_id = create_endpoint(client, branch="v1")
    update_endpoint(client, api_id, branch="v1")

    print("\nSetup complete. Run `python scripts/test_xano_match.py` to verify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
