"""ASGI entrypoint — UI + API."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import __version__
from app.api import router as api_router
from app.cad.dwg_convert import detect_engines
from app.config import get_settings

settings = get_settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.results_dir.mkdir(parents=True, exist_ok=True)

BASE = Path(__file__).parent
app = FastAPI(
    title="Image → Drawing",
    description="Векторизация архитектурных сканов в DXF/DWG (ГОСТ / СПДС)",
    version=__version__,
)
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))
app.include_router(api_router)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "ai_mode": settings.ai_mode,
        "env": settings.app_env,
        "dwg": detect_engines(),
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"version": __version__, "ai_mode": settings.ai_mode},
    )
