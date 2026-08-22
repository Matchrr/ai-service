"""Agent C: cosine similarity + LLM Fit Scorecard rerank."""


class SemanticMatchAgent:
    name = "semantic_match"

    def run(self, candidate_id: str, limit: int = 10) -> dict:
        return {
            "agent": self.name,
            "candidate_id": candidate_id,
            "matches": [],
            "limit": limit,
        }
