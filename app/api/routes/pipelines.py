from fastapi import APIRouter
from pydantic import BaseModel

from app.pipelines.events import sync_events
from app.pipelines.jobs import sync_jobs

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


class SyncRequest(BaseModel):
    query: str
    location: str | None = None


@router.post("/jobs/sync")
def harvest_jobs(payload: SyncRequest) -> dict:
    return sync_jobs(payload.query, payload.location)


@router.post("/events/sync")
def harvest_events(payload: SyncRequest) -> dict:
    return sync_events(payload.query, payload.location)
