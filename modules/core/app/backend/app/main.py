import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers import (
    assistant,
    bionemo,
    bootstrap,
    docs,
    genomics,
    health,
    kermt,
    large_molecule,
    me,
    models,
    monitoring,
    profile,
    settings,
    single_cell,
    small_molecule,
)
from app.services.workbench import initialize_lib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        initialize_lib()
        logger.info("Genesis Workbench lib initialized")
    except Exception as e:
        logger.warning("Lib initialization failed at startup: %s. Endpoints that need it will fail.", e)
    yield


app = FastAPI(
    title="Genesis Workbench",
    description="Genesis Workbench backend API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(me.router)
app.include_router(bootstrap.router)
app.include_router(profile.router)
app.include_router(settings.router)
app.include_router(monitoring.router)
app.include_router(assistant.router)
app.include_router(docs.router)
app.include_router(models.router)
app.include_router(large_molecule.router)
app.include_router(single_cell.router)
app.include_router(small_molecule.router)
app.include_router(genomics.router)
app.include_router(bionemo.router)
app.include_router(kermt.router)

FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
