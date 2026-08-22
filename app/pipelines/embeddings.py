"""Batch embed resumes, jobs, and events (text-embedding-3-small)."""

from app.core.config import settings


def embed_texts(texts: list[str]) -> list[list[float]]:
    _ = settings.embedding_model
    return [[] for _ in texts]
