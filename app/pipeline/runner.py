"""Pipeline runner — ТЗ stages 1–5 + DXF package."""

from __future__ import annotations

import json
import time
import traceback

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
        jobs.update(job_id, status=JobStatus.running, progress=5, step="Подготовка листов…")
        from app.pipeline.pdf_raster import expand_drawing_inputs

        sheet_paths = expand_drawing_inputs(job.drawing_paths, out_root)
        if not sheet_paths:
            raise ValueError("Нет растровых листов для обработки (проверьте PDF/изображения)")

        jobs.update(job_id, progress=12, step="Очистка сканов…")
        from app.pipeline.denoise import denoise_image

        cleaned_dir = out_root / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        cleaned = []
        for path in sheet_paths:
            out = cleaned_dir / f"{path.stem}_clean.png"
            denoise_image(path, out)
            cleaned.append(out)

        jobs.update(job_id, progress=28, step="Распознавание геометрии / OCR…")
        from app.pipeline.geometry import extract_geometry

        geometry = extract_geometry(
            cleaned,
            ai_mode=settings.ai_mode,
            source_images=sheet_paths,
            vectorization_profile=str(job.metadata.get("vectorization_profile") or settings.vectorization_profile),
            vectorization_auto_profile=bool(settings.vectorization_auto_profile),
        )
        needs_scale = bool(geometry.get("needs_scale")) and not job.metadata.get("scale_mm")
        jobs.update(
            job_id,
            needs_scale=needs_scale,
            scale_warning=geometry.get("scale_warning") if needs_scale else None,
        )

        scale_mm = job.metadata.get("scale_mm")
        if scale_mm:
            try:
                target = float(scale_mm)
                dims = geometry.get("dimensions") or []
                if dims and dims[0].get("value_mm"):
                    factor = target / float(dims[0]["value_mm"])
                    for key in ("walls", "partitions", "details", "strokes", "hatches"):
                        for poly in geometry.get(key) or []:
                            for p in poly:
                                p[0] *= factor
                                p[1] *= factor
                    for op in geometry.get("openings") or []:
                        for k in ("p1", "p2"):
                            if k in op:
                                op[k][0] *= factor
                                op[k][1] *= factor
                        if op.get("arc"):
                            for k in ("cx", "cy", "r"):
                                if k in op["arc"]:
                                    op["arc"][k] = float(op["arc"][k]) * factor
                    for circ in geometry.get("circles") or []:
                        for k in ("cx", "cy", "r"):
                            if k in circ:
                                circ[k] = float(circ[k]) * factor
                    for axis in geometry.get("axes") or []:
                        for k, v in list(axis.items()):
                            if k in {"name", "sheet", "ocr"}:
                                continue
                            if k == "bubble" and isinstance(v, dict):
                                for bk in ("cx", "cy", "r"):
                                    if bk in v:
                                        v[bk] = float(v[bk]) * factor
                                continue
                            if isinstance(v, (int, float)):
                                axis[k] = float(v) * factor
                    for t in geometry.get("texts") or []:
                        if t.get("pos"):
                            t["pos"] = [t["pos"][0] * factor, t["pos"][1] * factor]
                        if t.get("height"):
                            t["height"] = float(t["height"]) * factor
                    for d in dims:
                        d["value_mm"] = round(float(d["value_mm"]) * factor)
                        if "p1" in d and "p2" in d:
                            d["p1"] = [d["p1"][0] * factor, d["p1"][1] * factor]
                            d["p2"] = [d["p2"][0] * factor, d["p2"][1] * factor]
                    geometry["scale_factor"] = factor
                    geometry["needs_scale"] = False
                    geometry["scale_warning"] = None
                    geometry["scale_source"] = "user"
                    geometry["scale_confidence"] = max(float(geometry.get("scale_confidence") or 0), 0.9)
                    jobs.update(job_id, needs_scale=False, scale_warning=None)
            except (TypeError, ValueError):
                pass

        jobs.update(job_id, progress=48, step="Фотограмметрия / ORB alignment…")
        if job.photo_paths:
            from app.pipeline.photogrammetry import enrich_from_photos

            plan_ref = sheet_paths[0] if sheet_paths else None
            geometry = enrich_from_photos(
                geometry,
                job.photo_paths,
                ai_mode=settings.ai_mode,
                results_dir=out_root,
                plan_path=plan_ref,
            )

        jobs.update(job_id, progress=62, step="Кросс-валидация размеров…")
        from app.pipeline.validate import correlate_dimensions

        geometry = correlate_dimensions(geometry, metadata=job.metadata)
        geometry_path = out_root / "geometry.json"
        geometry_path.write_text(
            json.dumps(geometry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Live preview JSON (no DXF yet) — filters apply client-side
        jobs.update(job_id, progress=82, step="Подготовка интерактивного предпросмотра…")
        from app.pipeline.preview import build_preview_payload

        preview = build_preview_payload(geometry)
        preview_path = out_root / "preview.json"
        preview_path.write_text(
            json.dumps(preview, ensure_ascii=False),
            encoding="utf-8",
        )

        jobs.update(
            job_id,
            status=JobStatus.done,
            progress=100,
            step="Предпросмотр готов — настройте фильтры и сгенерируйте DXF",
            geometry_path=geometry_path,
            preview_path=preview_path,
            result_path=None,
            dxf_path=None,
            package_path=None,
            scene_path=None,
        )
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
