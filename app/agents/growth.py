"""Agent F: skill-gap growth plan — courses, YouTube, certs. No grammar nits."""


class CareerGrowthAdvisor:
    name = "career_growth"

    def run(self, candidate_id: str, target_title: str | None = None) -> dict:
        return {
            "agent": self.name,
            "candidate_id": candidate_id,
            "target_title": target_title,
            "skill_gaps": [],
        }
