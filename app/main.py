"""ASGI entrypoint — health + placeholder home for Docker bootstrap."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import get_settings

settings = get_settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.results_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Image → Drawing",
    description="Векторизация архитектурных сканов в DXF/DWG (ГОСТ / СПДС)",
    version=__version__,
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "ai_mode": settings.ai_mode,
        "env": settings.app_env,
    }


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Image → Drawing</title>
  <style>
    :root {{
      --bg: #0f1419;
      --fg: #e7ecf1;
      --muted: #8b9aab;
      --accent: #3d8bfd;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; min-height: 100vh; font-family: "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(1200px 600px at 10% -10%, #1a2740, var(--bg));
      color: var(--fg); display: grid; place-items: center; padding: 1.5rem;
    }}
    main {{ max-width: 36rem; text-align: center; }}
    h1 {{ font-weight: 650; letter-spacing: -0.02em; margin: 0 0 0.5rem; }}
    p {{ color: var(--muted); line-height: 1.5; }}
    code {{ color: var(--accent); }}
  </style>
</head>
<body>
  <main>
    <h1>Image → Drawing</h1>
    <p>Docker bootstrap готов. UI загрузки и пайплайн появятся в следующих коммитах.</p>
    <p>v{__version__} · AI mode: <code>{settings.ai_mode}</code></p>
  </main>
</body>
</html>"""
