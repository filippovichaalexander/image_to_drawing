"""Pipeline runner — orchestrates stages; heavy CV behind AI_MODE."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from app.config import get_settings
from app.jobs import JobStatus, jobs


def run_pipeline(job_id: str) -> None:
    """Background entrypoint. Stages filled in subsequent commits."""
    job = jobs.get(job_id)
    if not job:
        return

    settings = get_settings()
    try:
        jobs.update(job_id, status=JobStatus.running, progress=10, step="Очистка сканов…")
        time.sleep(0.3)

        # Lazy imports keep cold-start light when only checking /health
        from app.pipeline.denoise import denoise_image

        cleaned_dir = settings.results_dir / job_id / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        cleaned: list[Path] = []
        for path in job.drawing_paths:
            if path.suffix.lower() == ".pdf":
                # PDF rasterization — post-MVP; skip with note in stub mode
                continue
            out = cleaned_dir / f"{path.stem}_clean.png"
            denoise_image(path, out)
            cleaned.append(out)

        jobs.update(job_id, progress=35, step="Распознавание геометрии…")
        time.sleep(0.2)
        from app.pipeline.geometry import extract_geometry

        geometry = extract_geometry(cleaned, ai_mode=settings.ai_mode)

        jobs.update(job_id, progress=55, step="Фотограмметрия (если есть)…")
        time.sleep(0.15)
        if job.photo_paths:
            from app.pipeline.photogrammetry import enrich_from_photos

            geometry = enrich_from_photos(geometry, job.photo_paths, ai_mode=settings.ai_mode)

        jobs.update(job_id, progress=70, step="Кросс-валидация размеров…")
        time.sleep(0.15)
        from app.pipeline.validate import correlate_dimensions

        geometry = correlate_dimensions(geometry)

        jobs.update(job_id, progress=85, step="Генерация DXF по ГОСТ…")
        from app.cad.export import build_dxf

        out_dxf = settings.results_dir / job_id / f"{job.metadata.get('code', 'drawing')}.dxf"
        out_dxf.parent.mkdir(parents=True, exist_ok=True)
        build_dxf(geometry, job.metadata, out_dxf)

        jobs.update(
            job_id,
            status=JobStatus.done,
            progress=100,
            step="Готово",
            result_path=out_dxf,
        )
    except Exception as exc:  # noqa: BLE001 — surface to job status
        jobs.update(
            job_id,
            status=JobStatus.error,
            error=str(exc),
            step="Ошибка",
            progress=100,
        )
        traceback.print_exc()
