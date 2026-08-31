"""Discover Xano workspace and API group IDs using the Metadata API."""
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


def _get(path: str, token: str, params: dict | None = None) -> dict:
    url = f"{_meta_url()}{path}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = httpx.get(url, headers=headers, params=params, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    print(f"  [GET {path} -> {type(data).__name__} keys={list(data.keys()) if isinstance(data, dict) else len(data) if isinstance(data, list) else ''}]")
    return data


def main() -> int:
    _load_dotenv()
    token = os.environ.get("XANO_METADATA_API_TOKEN", "").strip()
    if not token:
        print("ERROR: XANO_METADATA_API_TOKEN not set")
        return 1

    print(f"Metadata API: {_meta_url()}")
    print(f"Token present: {bool(token)}\n")

    # List workspaces
    print("Workspaces:")
    try:
        workspaces = _get("/workspace", token)
        for ws in workspaces if isinstance(workspaces, list) else workspaces.get("items", []):
            print(f"  id={ws.get('id')} name={ws.get('name')}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # List API groups in workspace 1
    print("\nAPI groups in workspace 1:")
    try:
        groups = _get("/workspace/1/apigroup", token)
        for g in groups if isinstance(groups, list) else groups.get("items", []):
            print(f"  id={g.get('id')} name={g.get('name')} slug={g.get('slug')} prefix={g.get('prefix')}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # List custom functions in workspace 1 (paginated)
    print("\nCustom functions in workspace 1:")
    try:
        page = 1
        while True:
            funcs = _get("/workspace/1/function", token, params={"page": page, "per_page": 100})
            items = funcs.get("items", [])
            for f in items:
                print(f"  id={f.get('id')} name={f.get('name')}")
            if not funcs.get("nextPage") or not items:
                break
            page += 1
    except Exception as e:
        print(f"  ERROR: {e}")

    # List endpoints in all API groups
    print("\nEndpoints in all API groups:")
    try:
        groups = _get("/workspace/1/apigroup", token)
        for g in groups.get("items", []):
            g_id = g.get("id")
            g_name = g.get("name")
            page = 1
            while True:
                endpoints = _get(f"/workspace/1/apigroup/{g_id}/api", token, params={"page": page, "per_page": 100})
                items = endpoints.get("items", [])
                for ep in items:
                    print(f"  group={g_name} id={ep.get('id')} verb={ep.get('verb')} path={ep.get('name')} description={ep.get('description')}")
                if not endpoints.get("nextPage") or not items:
                    break
                page += 1
    except Exception as e:
        print(f"  ERROR: {e}")

    # List tables in workspace 1
    print("\nTables in workspace 1:")
    try:
        tables = _get("/workspace/1/table", token, params={"per_page": 100})
        items = tables.get("items", []) if isinstance(tables, dict) else tables
        for t in items:
            print(f"  id={t.get('id')} name={t.get('name')}")
    except Exception as e:
        print(f"  ERROR: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
