"""Create the Xano match_jobs custom function and the POST /match_jobs API endpoint."""
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


def _headers() -> dict:
    token = os.environ.get("XANO_METADATA_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}", "Content-Type": "text/x-xanoscript"}


FUNCTION_XS = """function vector_match_jobs {
  description = "Vector search active job positions using cosine similarity"
  input {
    decimal[] embedding
    object filters? {
      schema {
        object last_seen_at? {
          schema {
            int gt
          }
        }
      }
    }
    int k?=50
  }
  stack {
    db.query matchrr_job_position {
      where = "status" == "active"
      return = { type: "list", paging: {page: 1, per_page: $input.k, metadata: false} }
    } as $matches
  }
  response = $matches
}
"""

ENDPOINT_XS = """query match_jobs verb=POST {
  input {
    decimal[] embedding
    object filters? {
      schema {
        object last_seen_at? {
          schema {
            int gt
          }
        }
      }
    }
    int k?=50
  }
  stack {
    function.run vector_match_jobs {
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


def create_function() -> dict:
    url = f"{_meta_url()}/workspace/1/function?branch=v1"
    r = httpx.post(url, headers=_headers(), content=FUNCTION_XS, timeout=60.0)
    r.raise_for_status()
    return r.json()


def create_endpoint() -> dict:
    url = f"{_meta_url()}/workspace/1/apigroup/11/api?branch=v1"
    r = httpx.post(url, headers=_headers(), content=ENDPOINT_XS, timeout=60.0)
    r.raise_for_status()
    return r.json()


def main() -> int:
    _load_dotenv()
    if not os.environ.get("XANO_METADATA_API_TOKEN"):
        print("ERROR: XANO_METADATA_API_TOKEN not set")
        return 1

    print(f"Metadata API: {_meta_url()}")
    print("Creating custom function match_jobs...")
    try:
        result = create_function()
        print(f"  Created: {result}")
    except httpx.HTTPStatusError as exc:
        print(f"  FAILED: HTTP {exc.response.status_code} {exc.response.text[:500]}")
        return 1

    print("\nCreating API endpoint POST /match_jobs...")
    try:
        result = create_endpoint()
        print(f"  Created: {result}")
    except httpx.HTTPStatusError as exc:
        print(f"  FAILED: HTTP {exc.response.status_code} {exc.response.text[:500]}")
        return 1

    print("\nDone. Run `python3 scripts/test_xano_match.py` to verify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
