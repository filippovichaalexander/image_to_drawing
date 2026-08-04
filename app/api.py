"""HTTP API: jobs, scene for WebGL, DXF download."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

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


@router.get("/capabilities")
async def capabilities() -> dict[str, Any]:
    settings = get_settings()
    from app.pipeline.drawing_type import DRAWING_TYPES

    return {
        "ai_mode": settings.ai_mode,
        "vectorization_profile": settings.vectorization_profile,
        "vectorization_auto_profile": settings.vectorization_auto_profile,
        "use_sam": settings.use_sam,
        "export_format": "dxf",
        "drawing_types": [
            {"id": k, "label": v["label_ru"], "note": v["note"]}
            for k, v in DRAWING_TYPES.items()
        ],
    }


@router.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    drawings: Annotated[list[UploadFile], File()],
    photos: Annotated[list[UploadFile] | None, File()] = None,
    object_name: Annotated[str, Form()] = "",
    code: Annotated[str, Form()] = "",
    sheet_name: Annotated[str, Form()] = "План",
    stage: Annotated[str, Form()] = "Р",
    drawn_by: Annotated[str, Form()] = "",
    checked_by: Annotated[str, Form()] = "",
    company: Annotated[str, Form()] = "Мосреставрация",
    scale_mm: Annotated[str | None, Form()] = None,
    vectorization_profile: Annotated[str | None, Form()] = None,
    use_sam: Annotated[str | None, Form()] = None,
    drawing_type: Annotated[str | None, Form()] = None,
    drawing_note: Annotated[str | None, Form()] = None,
) -> dict:
    if not drawings:
        raise HTTPException(400, detail="Нужен хотя бы один чертёж")

    settings = get_settings()
    sam_flag = str(use_sam or "").lower() in {"1", "true", "on", "yes"}
    from app.pipeline.drawing_type import resolve_drawing_type

    dtype = resolve_drawing_type(drawing_type, drawing_note)
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
            "vectorization_profile": vectorization_profile or settings.vectorization_profile,
            "use_sam": sam_flag if use_sam is not None else bool(settings.use_sam),
            **dtype,
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
    public = jobs.get(job.id).to_public()  # type: ignore[union-attr]
    return {"job_id": job.id, "needs_scale": False, **public}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, detail="Задача не найдена")
    return job.to_public()


@router.get("/jobs/{job_id}/scene")
async def get_scene(job_id: str) -> JSONResponse:
    job = jobs.get(job_id)
    if not job or not job.scene_path or not job.scene_path.exists():
        raise HTTPException(404, detail="Сцена ещё не готова")
    data = json.loads(job.scene_path.read_text(encoding="utf-8"))
    return JSONResponse(data)


@router.get("/jobs/{job_id}/raw")
async def get_raw_geometry(job_id: str) -> JSONResponse:
    """Raw contours + metrics for live preview filters (no DXF required)."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, detail="Задача не найдена")
    if job.preview_path and job.preview_path.exists():
        data = json.loads(job.preview_path.read_text(encoding="utf-8"))
        if "scan_url" not in data:
            data["scan_url"] = f"/api/jobs/{job_id}/scan"
        return JSONResponse(data)
    if job.geometry_path and job.geometry_path.exists():
        from app.pipeline.preview import build_preview_payload

        geometry = json.loads(job.geometry_path.read_text(encoding="utf-8"))
        data = build_preview_payload(geometry)
        data["scan_url"] = f"/api/jobs/{job_id}/scan"
        return JSONResponse(data)
    raise HTTPException(404, detail="Предпросмотр ещё не готов")


@router.get("/jobs/{job_id}/scan")
async def get_job_scan(job_id: str) -> FileResponse:
    """Original first drawing sheet for preview underlay compare."""
    job = jobs.get(job_id)
    if not job or not job.drawing_paths:
        raise HTTPException(404, detail="Скан не найден")
    path = job.drawing_paths[0]
    if not path.exists():
        raise HTTPException(404, detail="Файл скана отсутствует")
    media = "image/jpeg"
    suf = path.suffix.lower()
    if suf == ".png":
        media = "image/png"
    elif suf in {".tif", ".tiff"}:
        media = "image/tiff"
    elif suf == ".pdf":
        # Prefer rasterized sheet from results if present
        settings = get_settings()
        raster = settings.results_dir / job_id / "sheets"
        candidates = sorted(raster.glob("*.png")) + sorted(raster.glob("*.jpg"))
        if candidates:
            path = candidates[0]
            media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        else:
            raise HTTPException(404, detail="PDF ещё не растеризован для подложки")
    return FileResponse(path, media_type=media, filename=path.name)


@router.post("/jobs/{job_id}/export")
async def export_with_filters(
    job_id: str,
    body: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    """Bake current live filters into DXF (server-side, from cached geometry)."""
    job = jobs.get(job_id)
    if not job or not job.geometry_path or not job.geometry_path.exists():
        raise HTTPException(404, detail="Геометрия ещё не готова")

    settings = get_settings()
    out_root = settings.results_dir / job_id
    out_root.mkdir(parents=True, exist_ok=True)

    geometry = json.loads(job.geometry_path.read_text(encoding="utf-8"))
    preview = None
    if job.preview_path and job.preview_path.exists():
        preview = json.loads(job.preview_path.read_text(encoding="utf-8"))

    filters = body.get("filters") if isinstance(body, dict) else None
    from app.pipeline.preview import apply_filters_to_geometry

    filtered = apply_filters_to_geometry(geometry, filters=filters, preview=preview)

    from app.cad.export import build_dxf
    from app.cad.package import build_download_package
    from app.cad.scene import dxf_to_scene

    code = str(job.metadata.get("code", "drawing")).replace("/", "-")
    out_dxf = out_root / f"{code}.dxf"
    build_dxf(filtered, job.metadata, out_dxf)

    package_path = out_root / f"{code}_package.zip"
    build_download_package(out_dxf, package_path, job.metadata)
    scene = dxf_to_scene(out_dxf)
    scene_path = out_root / "scene.json"
    scene_path.write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")

    (out_root / "export_filters.json").write_text(
        json.dumps(filters or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    jobs.update(
        job_id,
        result_path=out_dxf,
        dxf_path=out_dxf,
        package_path=package_path,
        scene_path=scene_path,
        step="DXF готов",
    )
    return {
        "status": "ok",
        "job_id": job_id,
        "wall_count": len(filtered.get("walls") or []),
        "download": f"/api/jobs/{job_id}/download",
        "has_dxf": True,
    }


@router.get("/jobs/{job_id}/download")
async def download_job(job_id: str) -> FileResponse:
    """Primary download: DXF file."""
    job = jobs.get(job_id)
    if not job or not job.dxf_path or not job.dxf_path.exists():
        raise HTTPException(404, detail="DXF ещё не готов — сначала нажмите «Сгенерировать DXF»")
    return FileResponse(
        path=job.dxf_path,
        filename=job.dxf_path.name,
        media_type="application/dxf",
    )


@router.get("/jobs/{job_id}/dxf")
async def download_dxf(job_id: str) -> FileResponse:
    return await download_job(job_id)


@router.get("/jobs/{job_id}/package")
async def download_package(job_id: str) -> FileResponse:
    """Optional ZIP (DXF + README)."""
    job = jobs.get(job_id)
    path = job.package_path if job else None
    if not job or not path or not path.exists():
        raise HTTPException(404, detail="Пакет ещё не готов")
    return FileResponse(
        path=path,
        filename=path.name,
        media_type="application/zip",
    )
