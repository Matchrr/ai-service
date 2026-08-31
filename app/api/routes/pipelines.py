from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.pipelines.cover_letters import generate_cover_letter
from app.pipelines.events import sync_events
from app.pipelines.jobs import expire_jobs, fanout_jobs, standing_harvest, sync_jobs
from app.pipelines.profile_chunks import upsert_profile_chunks
from app.schemas.profile import ProfilePayload

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


class SyncRequest(BaseModel):
    query: str
    location: str | None = None


class FanoutRequest(BaseModel):
    target_title: str
    location: str | None = None
    top_skill: str | None = None
    work_modes: list[str] | None = None
    desired_roles: list[str] | None = None


class ChunkSyncRequest(BaseModel):
    user_id: int
    profile: ProfilePayload
    profile_revision_id: int | None = None


def _require_service_secret(secret: str | None) -> None:
    expected = settings.matchr_service_secret
    if not expected:
        return
    if (secret or "") != expected:
        raise HTTPException(status_code=403, detail="Forbidden")


@router.post("/jobs/sync")
def harvest_jobs(payload: SyncRequest) -> dict:
    return sync_jobs(payload.query, payload.location)


@router.post("/jobs/fanout")
def harvest_fanout(payload: FanoutRequest) -> dict:
    try:
        return fanout_jobs(
            payload.target_title,
            payload.location,
            payload.top_skill,
            payload.work_modes,
            payload.desired_roles,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/jobs/standing")
def harvest_standing(
    x_matchr_service_secret: str | None = Header(default=None, alias="X-Matchr-Service-Secret"),
) -> dict:
    _require_service_secret(x_matchr_service_secret)
    return standing_harvest()


@router.post("/jobs/expire")
def harvest_expire(
    x_matchr_service_secret: str | None = Header(default=None, alias="X-Matchr-Service-Secret"),
) -> dict:
    _require_service_secret(x_matchr_service_secret)
    return expire_jobs()


@router.post("/events/sync")
def harvest_events(payload: SyncRequest) -> dict:
    return sync_events(payload.query, payload.location)


@router.post("/profile/chunks")
def sync_profile_chunks(payload: ChunkSyncRequest) -> dict:
    stats = upsert_profile_chunks(payload.user_id, payload.profile.model_dump())
    return {"source": "xano", "profile_revision_id": payload.profile_revision_id, **stats}


@router.post("/cover-letters/generate")
def pipeline_generate_cover_letter(payload: dict) -> dict:
    return generate_cover_letter(payload)
