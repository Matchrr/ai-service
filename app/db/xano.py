"""HTTP client for Xano job, profile-chunk, and cover-letter tables."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class XanoError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.xano_api_key:
        headers["Authorization"] = f"Bearer {settings.xano_api_key}"
    return headers


def _unwrap(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("items", "payload", "records", "result"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


class XanoClient:
    def __init__(self) -> None:
        self.base = settings.xano_api_url.rstrip("/") + "/"
        self.jobs_table = settings.xano_jobs_table.strip("/")
        self.runs_table = settings.xano_search_run_table.strip("/")
        self.chunks_table = settings.xano_chunks_table.strip("/")
        self.cover_letters_table = settings.xano_cover_letters_table.strip("/")

    @property
    def configured(self) -> bool:
        return bool(settings.xano_api_url)

    def _url(self, path: str) -> str:
        return urljoin(self.base, path.lstrip("/"))

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if not self.configured:
            raise XanoError("XANO_API_URL is not set")
        try:
            response = httpx.request(
                method,
                self._url(path),
                headers=_headers(),
                timeout=_TIMEOUT,
                **kwargs,
            )
        except httpx.HTTPError as exc:
            raise XanoError(f"Xano {method} {path} failed: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text[:400]
            raise XanoError(f"Xano {method} {path} → {response.status_code}: {detail}")
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return None

    def list_records(self, table: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page = 1
        per_page = 100
        while page <= 50:
            query = {"page": page, "per_page": per_page, **(params or {})}
            data = self._request("GET", table, params=query)
            batch = _unwrap(data)
            if not batch:
                break
            existing = {row.get("id") for row in items}
            if page > 1 and all(row.get("id") in existing for row in batch):
                break
            items.extend(batch)
            if len(batch) < per_page:
                break
            page += 1
        return items

    def get(self, table: str, record_id: int | str) -> dict[str, Any] | None:
        data = self._request("GET", f"{table}/{record_id}")
        return data if isinstance(data, dict) else None

    def add(self, table: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = self._request("POST", table, json=payload)
        if not isinstance(data, dict):
            raise XanoError(f"Xano POST {table} returned no record")
        return data

    def patch(self, table: str, record_id: int | str, payload: dict[str, Any]) -> dict[str, Any]:
        data = self._request("PATCH", f"{table}/{record_id}", json=payload)
        return data if isinstance(data, dict) else {"id": record_id, **payload}

    def delete(self, table: str, record_id: int | str) -> None:
        self._request("DELETE", f"{table}/{record_id}")

    def list_jobs(self, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self.list_records(self.jobs_table, params)

    def vector_search_jobs(
        self,
        embedding: list[float],
        filters: dict[str, Any] | None = None,
        k: int = 50,
    ) -> list[dict[str, Any]] | None:
        """Call a Xano API endpoint that performs vector search on the jobs table.

        The endpoint should be a published API endpoint whose function stack runs
        the vector-search custom function (e.g., match_jobs). Xano custom functions
        cannot be invoked directly over HTTP; they must be exposed through an API
        endpoint.

        Returns None if no endpoint is configured or if the call fails, so the caller
        can fall back to brute-force cosine over the active pool.
        """
        endpoint = (settings.xano_match_endpoint or "").strip()
        if not endpoint:
            return None
        payload = {
            "embedding": embedding,
            "filters": filters or {},
            "k": k,
        }
        try:
            data = self._request("POST", endpoint, json=payload)
        except XanoError:
            logger.exception("Xano vector search endpoint failed")
            return None
        return _unwrap(data)

    def add_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.add(self.jobs_table, payload)

    def patch_job(self, record_id: int | str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.patch(self.jobs_table, record_id, payload)

    def delete_job(self, record_id: int | str) -> None:
        self.delete(self.jobs_table, record_id)

    def list_runs(self) -> list[dict[str, Any]]:
        try:
            return self.list_records(self.runs_table)
        except XanoError as exc:
            logger.warning("Search-run table unavailable: %s", exc)
            return []

    def add_run(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return self.add(self.runs_table, payload)
        except XanoError as exc:
            logger.warning("Could not write search-run: %s", exc)
            return None

    def patch_run(self, record_id: int | str, payload: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return self.patch(self.runs_table, record_id, payload)
        except XanoError as exc:
            logger.warning("Could not patch search-run %s: %s", record_id, exc)
            return None

    def delete_run(self, record_id: int | str) -> None:
        try:
            self.delete(self.runs_table, record_id)
        except XanoError as exc:
            logger.warning("Could not delete search-run %s: %s", record_id, exc)

    def list_chunks(self, user_id: int) -> list[dict[str, Any]]:
        try:
            rows = self.list_records(self.chunks_table, {"user_id": user_id})
        except XanoError:
            rows = self.list_records(self.chunks_table)
        return [row for row in rows if _as_int(row.get("user_id")) == user_id]

    def vector_search_chunks(
        self,
        embedding: list[float],
        user_id: int,
        k: int = 8,
    ) -> list[dict[str, Any]] | None:
        """Nearest active chunks for this user. None means caller should brute-force."""
        function_name = (settings.xano_chunk_match_function or "").strip()
        if not function_name:
            return None
        payload = {
            "embedding": embedding,
            "filters": {"user_id": user_id, "is_active": True},
            "k": k,
        }
        try:
            data = self._request("POST", function_name, json=payload)
        except XanoError:
            logger.exception("Xano chunk vector search failed")
            return None
        rows = _unwrap(data)
        return [row for row in rows if _as_int(row.get("user_id")) == user_id]

    def add_chunk(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.add(self.chunks_table, payload)

    def patch_chunk(self, record_id: int | str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.patch(self.chunks_table, record_id, payload)

    def list_cover_letters(self, user_id: int) -> list[dict[str, Any]]:
        try:
            rows = self.list_records(self.cover_letters_table, {"user_id": user_id})
        except XanoError:
            rows = self.list_records(self.cover_letters_table)
        return [row for row in rows if _as_int(row.get("user_id")) == user_id]

    def add_cover_letter(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.add(self.cover_letters_table, payload)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


xano = XanoClient()
