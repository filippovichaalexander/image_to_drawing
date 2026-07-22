"""HTTP API: create/list job status and download results."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.jobs import JobStatus, jobs
from app.pipeline.runner import run_pipeline

router = APIRouter(prefix="/api")

ALLOWED_DRAWING = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf"}
ALLOWED_PHOTO = {".jpg", ".jpeg", ".png", ".webp"}


def _safe_suffix(name: str) -> str:
    return Path(name).suffix.lower()


async def _save_uploads(
    files: list[UploadFile],
    dest_dir: Path,
    allowed: set[str],
    prefix: str,
) -> list[Path]:
    settings = get_settings()
    saved: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for idx, upload in enumerate(files):
        if not upload.filename:
            continue
        suffix = _safe_suffix(upload.filename)
        if suffix not in allowed:
            raise HTTPException(400, detail=f"Неподдерживаемый тип файла: {upload.filename}")
        target = dest_dir / f"{prefix}_{idx:02d}{suffix}"
        size = 0
        with target.open("wb") as out:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    out.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(400, detail=f"Файл слишком большой: {upload.filename}")
                out.write(chunk)
        saved.append(target)
    return saved


@router.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    drawings: Annotated[list[UploadFile], File()],
    photos: Annotated[list[UploadFile], File()] = [],
    object_name: Annotated[str, Form()] = "",
    code: Annotated[str, Form()] = "",
    sheet_name: Annotated[str, Form()] = "План",
    stage: Annotated[str, Form()] = "Р",
    drawn_by: Annotated[str, Form()] = "",
    checked_by: Annotated[str, Form()] = "",
    company: Annotated[str, Form()] = "Мосреставрация",
    scale_mm: Annotated[str | None, Form()] = None,
) -> dict:
    if not drawings:
        raise HTTPException(400, detail="Нужен хотя бы один чертёж")

    settings = get_settings()
    job = jobs.create(
        metadata={
            "object_name": object_name or "Объект",
            "code": code or "000-000-АР",
            "sheet_name": sheet_name,
            "stage": stage if stage in {"Р", "П"} else "Р",
            "drawn_by": drawn_by or "—",
            "checked_by": checked_by or "—",
            "company": company or "—",
            "sheet_num": "1",
            "scale_mm": scale_mm,
        }
    )
    job_dir = settings.upload_dir / job.id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    drawing_paths = await _save_uploads(drawings, job_dir / "drawings", ALLOWED_DRAWING, "d")
    photo_paths = await _save_uploads(photos or [], job_dir / "photos", ALLOWED_PHOTO, "p")
    if not drawing_paths:
        raise HTTPException(400, detail="Не удалось сохранить чертежи")

    jobs.update(
        job.id,
        drawing_paths=drawing_paths,
        photo_paths=photo_paths,
        status=JobStatus.queued,
        progress=5,
        step="Файлы загружены",
    )
    background_tasks.add_task(run_pipeline, job.id)
    return {"job_id": job.id, "needs_scale": False, **jobs.get(job.id).to_public()}  # type: ignore[union-attr]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, detail="Задача не найдена")
    return job.to_public()


@router.get("/jobs/{job_id}/download")
async def download_job(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job or not job.result_path or not job.result_path.exists():
        raise HTTPException(404, detail="Результат ещё не готов")
    return FileResponse(
        path=job.result_path,
        filename=job.result_path.name,
        media_type="application/dxf",
    )
