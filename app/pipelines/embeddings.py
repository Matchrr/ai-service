"""Batch embed resumes, jobs, and events via Gemini (cosine-friendly)."""

from __future__ import annotations

import logging
import math
import time

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)

_BATCH = 5
_MAX_CHARS = 6000
_NORMALIZED_DIMS = 3072


def _client() -> genai.Client | None:
    if not settings.embeddings_enabled:
        return None
    return genai.Client(api_key=settings.resolved_gemini_api_key)


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0:
        return list(values)
    return [v / norm for v in values]


def _maybe_normalize(values: list[float]) -> list[float]:
    if len(values) == _NORMALIZED_DIMS:
        return list(values)
    return _l2_normalize(values)


def _supports_task_type(model: str) -> bool:
    return "embedding-001" in model or "text-embedding-004" in model


def _embed_config(task_type: str | None) -> types.EmbedContentConfig:
    kwargs: dict[str, str | int] = {}
    dims = settings.embedding_dimensions
    if dims:
        kwargs["output_dimensionality"] = dims
    if task_type and _supports_task_type(settings.resolved_embedding_model):
        kwargs["task_type"] = task_type
    return types.EmbedContentConfig(**kwargs)


def _truncate(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_CHARS:
        return text
    return text[:_MAX_CHARS]


def _is_quota_error(exc: Exception) -> bool:
    message = str(exc)
    return "429" in message or "RESOURCE_EXHAUSTED" in message or "quota" in message.lower()


def embed_texts(texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed texts; empty strings stay empty vectors and are skipped in the API call."""
    vectors: list[list[float]] = [[] for _ in texts]
    pending = [(index, _truncate(text)) for index, text in enumerate(texts) if text and text.strip()]
    if not pending:
        return vectors

    client = _client()
    if client is None:
        logger.info("Embeddings skipped: no Gemini/Google credentials")
        return vectors

    for start in range(0, len(pending), _BATCH):
        chunk = pending[start : start + _BATCH]
        response = None
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                response = client.models.embed_content(
                    model=settings.resolved_embedding_model,
                    contents=[text for _, text in chunk],
                    config=_embed_config(task_type),
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if _is_quota_error(exc) and attempt < 5:
                    delay = min(32, 2 ** attempt)
                    logger.warning("Embedding 429/quota; retry in %ss", delay)
                    time.sleep(delay)
                    continue
                logger.warning("Embedding request failed: %s", exc)
                break
        if response is None:
            if last_error:
                logger.warning("Embedding chunk failed after retries: %s", last_error)
            continue
        items = list(response.embeddings or [])
        for offset, (index, _) in enumerate(chunk):
            if offset >= len(items) or items[offset] is None:
                continue
            values = list(items[offset].values or [])
            if values:
                vectors[index] = _maybe_normalize(values)
        if start + _BATCH < len(pending):
            time.sleep(0.4)
    return vectors


def embed_text(text: str, *, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    return embed_texts([text], task_type=task_type)[0]
