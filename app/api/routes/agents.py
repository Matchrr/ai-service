from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agents import (
    CareerGrowthAdvisor,
    EventMatchAgent,
    GroundingAuditor,
    OutreachDraftAgent,
    SemanticMatchAgent,
    TailoringAgent,
)

router = APIRouter(prefix="/agents", tags=["agents"])


class GroundRequest(BaseModel):
    linkedin_payload: dict | None = None
    resume_text: str | None = None


class CandidateJobRequest(BaseModel):
    candidate_id: str
    job_id: str | None = None
    target_title: str | None = None
    limit: int = 10


class OutreachRequest(BaseModel):
    recipient_email: str = Field(..., min_length=3)
    job_id: str | None = None
    extra_context: str | None = None


@router.post("/ground")
def ground_profile(payload: GroundRequest) -> dict:
    return GroundingAuditor().run(payload.linkedin_payload, payload.resume_text)


@router.post("/match")
def match_jobs(payload: CandidateJobRequest) -> dict:
    return SemanticMatchAgent().run(payload.candidate_id, payload.limit)


@router.post("/tailor")
def tailor_dossier(payload: CandidateJobRequest) -> dict:
    return TailoringAgent().run(payload.candidate_id, payload.job_id or "")


@router.post("/growth")
def growth_plan(payload: CandidateJobRequest) -> dict:
    return CareerGrowthAdvisor().run(payload.candidate_id, payload.target_title)


@router.post("/events")
def match_events(payload: CandidateJobRequest) -> dict:
    return EventMatchAgent().run(payload.candidate_id, payload.limit)


@router.post("/outreach")
def draft_outreach(payload: OutreachRequest) -> dict:
    return OutreachDraftAgent().run(payload.recipient_email, payload.job_id, payload.extra_context)
