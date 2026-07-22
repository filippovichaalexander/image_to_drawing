"""Pipeline runner — ТЗ stages 1–5 + DWG package."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from app.config import get_settings
from app.jobs import JobStatus, jobs


def run_pipeline(job_id: str) -> None:
    job = jobs.get(job_id)
    if not job:
        return

    settings = get_settings()
    out_root = settings.results_dir / job_id
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        jobs.update(job_id, status=JobStatus.running, progress=8, step="Очистка сканов…")
        from app.pipeline.denoise import denoise_image

        cleaned_dir = out_root / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        cleaned: list[Path] = []
        for path in job.drawing_paths:
            if path.suffix.lower() == ".pdf":
                continue
            out = cleaned_dir / f"{path.stem}_clean.png"
            denoise_image(path, out)
            cleaned.append(out)

        jobs.update(job_id, progress=28, step="Распознавание геометрии / OCR…")
        from app.pipeline.geometry import extract_geometry

        geometry = extract_geometry(cleaned, ai_mode=settings.ai_mode)

        # Optional user scale reference
        scale_mm = job.metadata.get("scale_mm")
        if scale_mm:
            try:
                target = float(scale_mm)
                dims = geometry.get("dimensions") or []
                if dims and dims[0].get("value_mm"):
                    factor = target / float(dims[0]["value_mm"])
                    for wall in geometry.get("walls") or []:
                        for p in wall:
                            p[0] *= factor
                            p[1] *= factor
                    for d in dims:
                        d["value_mm"] = round(float(d["value_mm"]) * factor)
                    geometry["scale_factor"] = factor
            except (TypeError, ValueError):
                pass

        jobs.update(job_id, progress=48, step="Фотограмметрия…")
        if job.photo_paths:
            from app.pipeline.photogrammetry import enrich_from_photos

            geometry = enrich_from_photos(
                geometry,
                job.photo_paths,
                ai_mode=settings.ai_mode,
                results_dir=out_root,
            )

        jobs.update(job_id, progress=62, step="Кросс-валидация размеров…")
        from app.pipeline.validate import correlate_dimensions

        geometry = correlate_dimensions(geometry)
        (out_root / "geometry.json").write_text(
            json.dumps(geometry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        jobs.update(job_id, progress=78, step="Генерация DXF по ГОСТ…")
        from app.cad.export import build_dxf

        code = str(job.metadata.get("code", "drawing")).replace("/", "-")
        out_dxf = out_root / f"{code}.dxf"
        build_dxf(geometry, job.metadata, out_dxf)

        jobs.update(job_id, progress=88, step="Конвертация DWG + пакет…")
        from app.cad.dwg_convert import build_download_package
        from app.cad.scene import dxf_to_scene

        package_path = out_root / f"{code}_package.zip"
        _, conv = build_download_package(out_dxf, package_path, job.metadata)
        scene = dxf_to_scene(out_dxf)
        (out_root / "scene.json").write_text(
            json.dumps(scene, ensure_ascii=False),
            encoding="utf-8",
        )

        jobs.update(
            job_id,
            status=JobStatus.done,
            progress=100,
            step="Готово",
            result_path=package_path,
            dxf_path=out_dxf,
            dwg_path=conv.dwg_path,
            dwg_engine=conv.engine,
            dwg_message=conv.message,
            scene_path=out_root / "scene.json",
        )
        # tiny pause so UI polls catch intermediate steps in fast stub runs
        time.sleep(0.05)
    except Exception as exc:  # noqa: BLE001
        jobs.update(
            job_id,
            status=JobStatus.error,
            error=str(exc),
            step="Ошибка",
            progress=100,
        )
        traceback.print_exc()
