"""Agent D: rewrite bullets against a job without inventing experience."""


class TailoringAgent:
    name = "tailor"

    def run(self, candidate_id: str, job_id: str) -> dict:
        return {
            "agent": self.name,
            "candidate_id": candidate_id,
            "job_id": job_id,
            "resume_bullets": [],
            "cover_letter": None,
        }
