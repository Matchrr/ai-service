"""Agent A: parse LinkedIn + resume into an immutable Ground Truth Profile."""


class GroundingAuditor:
    name = "grounding_auditor"

    def run(self, linkedin_payload: dict | None = None, resume_text: str | None = None) -> dict:
        source = "linkedin" if linkedin_payload else "resume" if resume_text else "empty"
        return {
            "agent": self.name,
            "source": source,
            "profile": {
                "headline": (linkedin_payload or {}).get("headline"),
                "skills": (linkedin_payload or {}).get("skills", []),
                "experience": (linkedin_payload or {}).get("experience", []),
            },
        }
