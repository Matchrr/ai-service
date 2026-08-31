"""Create Ground Truth Profile tables in Xano and publish CRUD in the Matchr API group.

Step 1 of Planning/cloud_data_storage.md: matchr_profile + fact children +
source_document. Login can later restore the profile from Xano instead of disk.

Uses XANO_METADATA_API_TOKEN in ai-service/.env. Idempotent: existing tables and
endpoints with the same name are left in place.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

AI_SERVICE = Path(__file__).resolve().parents[1]
WORKSPACE_ID = 1
APIGROUP_ID = 12  # Matchr (v1 live branch)
BRANCH = "v1"

TABLES: list[tuple[str, str, list[str]]] = [
    (
        "matchr_profile",
        """// Ground Truth Profile header: one row per user. Login hydrates from this row.
table matchr_profile {
  auth = false

  schema {
    int id
    timestamp created_at?=now {
      visibility = "private"
    }
    timestamp updated_at?

    // FK to Auth user. 1:1 — unique index below.
    int user_id

    text full_name? filters=trim
    text headline? filters=trim
    text summary? filters=trim
    // Resume email may differ from the login email.
    text email? filters=trim|lower
    text picture_url? filters=trim
    json websites?
    json skills?
    json languages?
    json honors?
    json projects?
    text target_title? filters=trim
    text location? filters=trim
    bool grounded?=false
    json grounding_sources?
    timestamp facts_accepted_at?
    bool linkedin_connected?=false
    text linkedin_member_id? filters=trim
    enum linkedin_coverage?=none {
      values = ["none", "identity", "profile"]
    }
    bool gmail_connected?=false
    vector? embedding? {
      size = 1536
    }
    // Accepted revision id. Int only — circular FK with matchr_profile_revision.
    int profile_revision_id?
  }

  index = [
    {type: "primary", field: [{name: "id"}]}
    {type: "btree", field: [{name: "created_at", op: "desc"}]}
    {type: "btree|unique", field: [{name: "user_id"}]}
    {type: "vector", field: [{name: "embedding", op: "vector_cosine_ops"}]}
  ]

  tags = ["matchr", "profile"]
}
""",
        [
            "user_id",
            "full_name",
            "headline",
            "summary",
            "email",
            "picture_url",
            "websites",
            "skills",
            "languages",
            "honors",
            "projects",
            "target_title",
            "location",
            "grounded",
            "grounding_sources",
            "facts_accepted_at",
            "linkedin_connected",
            "linkedin_member_id",
            "linkedin_coverage",
            "gmail_connected",
            "embedding",
            "profile_revision_id",
            "updated_at",
        ],
    ),
    (
        "matchr_experience",
        """// Normalized work / volunteer roles. Rewritten on each Accept.
table matchr_experience {
  auth = false

  schema {
    int id
    timestamp created_at?=now {
      visibility = "private"
    }
    int user_id
    int profile_id {
      table = "matchr_profile"
    }
    enum kind {
      values = ["work", "volunteer"]
    }
    text title? filters=trim
    text company? filters=trim
    text location? filters=trim
    text start_date? filters=trim
    text end_date? filters=trim
    json bullets?
    int sort_order?=0
  }

  index = [
    {type: "primary", field: [{name: "id"}]}
    {type: "btree", field: [{name: "created_at", op: "desc"}]}
    {type: "btree", field: [{name: "user_id"}]}
    {type: "btree", field: [{name: "profile_id"}]}
  ]

  tags = ["matchr", "profile"]
}
""",
        [
            "user_id",
            "profile_id",
            "kind",
            "title",
            "company",
            "location",
            "start_date",
            "end_date",
            "bullets",
            "sort_order",
        ],
    ),
    (
        "matchr_education",
        """// Education as display strings until a later school/degree split.
table matchr_education {
  auth = false

  schema {
    int id
    timestamp created_at?=now {
      visibility = "private"
    }
    int user_id
    int profile_id {
      table = "matchr_profile"
    }
    text school_text? filters=trim
    int sort_order?=0
  }

  index = [
    {type: "primary", field: [{name: "id"}]}
    {type: "btree", field: [{name: "created_at", op: "desc"}]}
    {type: "btree", field: [{name: "user_id"}]}
    {type: "btree", field: [{name: "profile_id"}]}
  ]

  tags = ["matchr", "profile"]
}
""",
        ["user_id", "profile_id", "school_text", "sort_order"],
    ),
    (
        "matchr_certification",
        """// Certifications as display strings.
table matchr_certification {
  auth = false

  schema {
    int id
    timestamp created_at?=now {
      visibility = "private"
    }
    int user_id
    int profile_id {
      table = "matchr_profile"
    }
    text cert_text? filters=trim
    int sort_order?=0
  }

  index = [
    {type: "primary", field: [{name: "id"}]}
    {type: "btree", field: [{name: "created_at", op: "desc"}]}
    {type: "btree", field: [{name: "user_id"}]}
    {type: "btree", field: [{name: "profile_id"}]}
  ]

  tags = ["matchr", "profile"]
}
""",
        ["user_id", "profile_id", "cert_text", "sort_order"],
    ),
    (
        "matchr_profile_revision",
        """// Immutable Accept snapshots. Dossiers cite grounded-as-of revision N.
table matchr_profile_revision {
  auth = false

  schema {
    int id
    timestamp created_at?=now {
      visibility = "private"
    }
    int user_id
    timestamp accepted_at?=now
    json sources?
    json snapshot?
  }

  index = [
    {type: "primary", field: [{name: "id"}]}
    {type: "btree", field: [{name: "created_at", op: "desc"}]}
    {type: "btree", field: [{name: "user_id"}]}
    {type: "btree", field: [{name: "accepted_at", op: "desc"}]}
  ]

  tags = ["matchr", "profile"]
}
""",
        ["user_id", "accepted_at", "sources", "snapshot"],
    ),
    (
        "matchr_source_document",
        """// Pointers to S3 source PDFs (resume / LinkedIn Save-to-PDF). Bytes stay in S3.
table matchr_source_document {
  auth = false

  schema {
    int id
    timestamp created_at?=now {
      visibility = "private"
    }
    int user_id
    enum kind {
      values = ["resume", "linkedin_pdf"]
    }
    text original_filename? filters=trim
    text content_type? filters=trim
    int byte_size?
    text sha256? filters=trim
    text s3_key? filters=trim
    bool nutrient_ok?=false
    text extracted_text_preview? filters=trim
    timestamp replaced_at?
  }

  index = [
    {type: "primary", field: [{name: "id"}]}
    {type: "btree", field: [{name: "created_at", op: "desc"}]}
    {type: "btree", field: [{name: "user_id"}]}
    {type: "btree", field: [{name: "sha256"}]}
  ]

  tags = ["matchr", "profile"]
}
""",
        [
            "user_id",
            "kind",
            "original_filename",
            "content_type",
            "byte_size",
            "sha256",
            "s3_key",
            "nutrient_ok",
            "extracted_text_preview",
            "replaced_at",
        ],
    ),
]


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


def _client() -> httpx.Client:
    token = os.environ.get("XANO_METADATA_API_TOKEN", "").strip()
    if not token:
        print("ERROR: XANO_METADATA_API_TOKEN is not set.")
        sys.exit(1)
    return httpx.Client(
        base_url=_meta_url(),
        headers={"Authorization": f"Bearer {token}"},
        timeout=60.0,
    )


def _items(data: object) -> list[dict]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        value = data.get("items")
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _existing_tables(client: httpx.Client) -> dict[str, int]:
    response = client.get(
        f"/workspace/{WORKSPACE_ID}/table",
        params={"per_page": 100},
    )
    response.raise_for_status()
    return {row["name"]: int(row["id"]) for row in _items(response.json()) if row.get("name")}


def _existing_endpoints(client: httpx.Client) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    page = 1
    while page <= 20:
        response = client.get(
            f"/workspace/{WORKSPACE_ID}/apigroup/{APIGROUP_ID}/api",
            params={"page": page, "per_page": 100, "branch": BRANCH},
        )
        response.raise_for_status()
        rows = _items(response.json())
        for row in rows:
            found.add((str(row.get("verb", "")).upper(), str(row.get("name", ""))))
        if len(rows) < 100:
            break
        page += 1
    return found


def _post_xs(client: httpx.Client, path: str, script: str, params: dict | None = None) -> httpx.Response:
    return client.post(
        path,
        headers={"Content-Type": "text/x-xanoscript"},
        content=script,
        params=params or {},
    )


def _crud_scripts(table: str, fields: list[str]) -> list[tuple[str, str, str]]:
    """Return (verb, name, xanoscript) for GET list / POST / GET / PATCH / DELETE."""
    id_param = f"{table}_id"
    data_lines = "\n".join(f"        {name}: $input.{name}" for name in fields)
    return [
        (
            "GET",
            table,
            f"""// Query all {table} records
query {table} verb=GET {{
  api_group = "Matchr"

  input {{
  }}

  stack {{
    db.query {table} {{
      return = {{type: "list"}}
    }} as $model
  }}

  response = $model
}}
""",
        ),
        (
            "POST",
            table,
            f"""// Add {table} record
query {table} verb=POST {{
  api_group = "Matchr"

  input {{
    dblink {{
      table = "{table}"
    }}
  }}

  stack {{
    db.add {table} {{
      enforce_hidden_fields = false
      data = {{
        created_at: "now"
{data_lines}
      }}
    }} as $model
  }}

  response = $model
}}
""",
        ),
        (
            "GET",
            f"{table}/{{{id_param}}}",
            f"""// Get {table} record
query "{table}/{{{id_param}}}" verb=GET {{
  api_group = "Matchr"

  input {{
    int {id_param}? filters=min:1
  }}

  stack {{
    db.get {table} {{
      field_name = "id"
      field_value = $input.{id_param}
    }} as $model

    precondition ($model != null) {{
      error_type = "notfound"
      error = "Not Found"
    }}
  }}

  response = $model
}}
""",
        ),
        (
            "PATCH",
            f"{table}/{{{id_param}}}",
            f"""// Edit {table} record
query "{table}/{{{id_param}}}" verb=PATCH {{
  api_group = "Matchr"

  input {{
    int {id_param}? filters=min:1
    dblink {{
      table = "{table}"
    }}
  }}

  stack {{
    util.get_raw_input {{
      encoding = "json"
      exclude_middleware = false
    }} as $raw_input

    db.patch {table} {{
      field_name = "id"
      field_value = $input.{id_param}
      data = `$input|pick:($raw_input|keys)`|filter_null|filter_empty_text
    }} as $model
  }}

  response = $model
}}
""",
        ),
        (
            "DELETE",
            f"{table}/{{{id_param}}}",
            f"""// Delete {table} record
query "{table}/{{{id_param}}}" verb=DELETE {{
  api_group = "Matchr"

  input {{
    int {id_param}? filters=min:1
  }}

  stack {{
    db.del {table} {{
      field_name = "id"
      field_value = $input.{id_param}
    }}
  }}

  response = null
}}
""",
        ),
    ]


def main() -> int:
    _load_dotenv()
    print(f"Metadata API: {_meta_url()}")
    print(f"Workspace {WORKSPACE_ID}, API group {APIGROUP_ID}, branch {BRANCH}\n")

    with _client() as client:
        tables = _existing_tables(client)
        print("Existing tables:", ", ".join(sorted(tables)) or "(none)")

        for name, script, _fields in TABLES:
            if name in tables:
                print(f"  skip table {name} (id={tables[name]})")
                continue
            print(f"  create table {name}...")
            response = _post_xs(
                client,
                f"/workspace/{WORKSPACE_ID}/table",
                script,
                params={"branch": BRANCH},
            )
            if response.status_code >= 400:
                print(f"    FAILED HTTP {response.status_code}: {response.text[:800]}")
                return 1
            created = response.json()
            table_id = created.get("id") if isinstance(created, dict) else None
            print(f"    created id={table_id}")
            if table_id is not None:
                tables[name] = int(table_id)

        endpoints = _existing_endpoints(client)
        print(f"\nExisting Matchr endpoints: {len(endpoints)}")
        for name, _script, fields in TABLES:
            for verb, path, xs in _crud_scripts(name, fields):
                key = (verb, path)
                if key in endpoints:
                    print(f"  skip {verb} /{path}")
                    continue
                print(f"  create {verb} /{path}...")
                response = _post_xs(
                    client,
                    f"/workspace/{WORKSPACE_ID}/apigroup/{APIGROUP_ID}/api",
                    xs,
                    params={"branch": BRANCH},
                )
                if response.status_code >= 400:
                    print(f"    FAILED HTTP {response.status_code}: {response.text[:800]}")
                    return 1
                created = response.json()
                ep_id = created.get("id") if isinstance(created, dict) else None
                print(f"    created id={ep_id}")
                endpoints.add(key)

    print("\nDone. Ground Truth Profile tables are live in Xano.")
    print("Next: S3 upload on resume/LinkedIn PDF, then stop writing profile_vault JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
