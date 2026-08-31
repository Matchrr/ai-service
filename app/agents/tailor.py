"""Agent D: RAG cover letter over grounded profile chunks, then persist."""

from __future__ import annotations

from typing import Any

from app.pipelines.cover_letters import generate_cover_letter


class TailoringAgent:
    name = "tailor"

    def run(self, payload: dict[str, Any]) -> dict:
        result = generate_cover_letter(payload)
        return {
            "agent": self.name,
            "candidate_id": payload.get("candidate_id"),
            "job_id": payload.get("job_id"),
            **result,
        }
