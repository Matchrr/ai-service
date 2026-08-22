from fastapi import APIRouter

from app.api.routes import agents, pipelines
from app.core.config import settings

api_router = APIRouter()
api_router.include_router(agents.router)
api_router.include_router(pipelines.router)


@api_router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
