from contextlib import asynccontextmanager
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings

logger = logging.getLogger(__name__)


def _run_standing_tick() -> None:
    try:
        from app.pipelines.jobs import standing_harvest

        stats = standing_harvest()
        logger.info("Standing harvest finished: %s", stats)
    except Exception:
        logger.exception("Standing harvest tick failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _run_standing_tick,
        IntervalTrigger(hours=max(1, settings.harvest_standing_hours)),
        id="standing_harvest",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    logger.info("Standing harvest scheduled every %sh", settings.harvest_standing_hours)
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.app_name, "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True)
