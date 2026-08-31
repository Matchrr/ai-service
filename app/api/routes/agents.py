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
from app.schemas.profile import ProfilePayload

router = APIRouter(prefix="/agents", tags=["agents"])


class GroundRequest(BaseModel):
    linkedin_payload: dict | None = None
    resume_text: str | None = None


class CandidateJobRequest(BaseModel):
    candidate_id: str
    job_id: str | None = None
    target_title: str | None = None
    desired_roles: list[str] = Field(default_factory=list)
    limit: int = 10
    profile_text: str = ""
    skills: list[str] = Field(default_factory=list)
    work_modes: list[str] | None = None
    job_types: list[str] | None = None
    pay_min: float | None = None
    pay_period: str | None = None
    location: str | None = None


class TailorRequest(BaseModel):
    candidate_id: str
    job_id: str | None = None
    user_id: int | None = None
    xano_job_id: int | None = None
    job_title: str = ""
    company: str = ""
    job_description: str = ""
    matching_skills: list[str] = Field(default_factory=list)
    missing_tech: list[str] = Field(default_factory=list)
    key_angle: str | None = None
    application_id: int | None = None
    profile_revision_id: int | None = None
    profile: ProfilePayload = Field(default_factory=ProfilePayload)


class OutreachRequest(BaseModel):
    recipient_email: str = Field(..., min_length=3)
    job_id: str | None = None
    extra_context: str | None = None


@router.post("/ground")
def ground_profile(payload: GroundRequest) -> dict:
    return GroundingAuditor().run(payload.linkedin_payload, payload.resume_text)


@router.post("/match")
def match_jobs(payload: CandidateJobRequest) -> dict:
    return SemanticMatchAgent().run(
        payload.candidate_id,
        payload.limit,
        target_title=payload.target_title,
        desired_roles=payload.desired_roles,
        profile_text=payload.profile_text,
        skills=payload.skills,
        work_modes=payload.work_modes,
        job_types=payload.job_types,
        pay_min=payload.pay_min,
        pay_period=payload.pay_period,
        location=payload.location,
    )


@router.post("/tailor")
def tailor_dossier(payload: TailorRequest) -> dict:
    return TailoringAgent().run(payload.model_dump())


@router.post("/growth")
def growth_plan(payload: CandidateJobRequest) -> dict:
    return CareerGrowthAdvisor().run(payload.candidate_id, payload.target_title)


@router.post("/events")
def match_events(payload: CandidateJobRequest) -> dict:
    return EventMatchAgent().run(payload.candidate_id, payload.limit)


@router.post("/outreach")
def draft_outreach(payload: OutreachRequest) -> dict:
    return OutreachDraftAgent().run(payload.recipient_email, payload.job_id, payload.extra_context)
