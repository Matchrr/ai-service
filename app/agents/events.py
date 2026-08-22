"""Agent G: harvest + rerank the most compatible networking events."""


class EventMatchAgent:
    name = "event_match"

    def run(self, candidate_id: str, limit: int = 8) -> dict:
        return {
            "agent": self.name,
            "candidate_id": candidate_id,
            "events": [],
            "limit": limit,
        }
