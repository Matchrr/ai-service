"""Agent E: Gmail-ready cold email; never sends on its own."""


class OutreachDraftAgent:
    name = "outreach_draft"

    def run(self, recipient_email: str, job_id: str | None = None, extra_context: str | None = None) -> dict:
        return {
            "agent": self.name,
            "recipient_email": recipient_email,
            "job_id": job_id,
            "subject": "Exploring a fit on your team",
            "body": extra_context or "Grounded draft will be written from the profile + Fit Scorecard.",
        }
