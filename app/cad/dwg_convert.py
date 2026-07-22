"""DXF → DWG conversion via LibreDWG (dxf2dwg) or ODA File Converter."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConvertResult:
    dwg_path: Path | None
    engine: str
    message: str


def _which(name: str) -> str | None:
    return shutil.which(name)


def detect_engines() -> dict[str, object]:
    oda = os.environ.get("ODA_FC_PATH") or _which("ODAFileConverter")
    return {
        "libredwg": bool(_which("dxf2dwg")),
        "oda": bool(oda),
        "oda_path": oda or "",
    }


def convert_dxf_to_dwg(dxf_path: Path, dwg_path: Path | None = None) -> ConvertResult:
    """
    Convert DXF to DWG.

    Preference: LibreDWG dxf2dwg (open-source, R2000) → ODA File Converter.
    LibreDWG currently encodes reliably up to R2000 — export DXF as R2000.
    """
    dxf_path = dxf_path.resolve()
    if not dxf_path.exists():
        return ConvertResult(None, "none", f"DXF not found: {dxf_path}")

    dwg_path = (dwg_path or dxf_path.with_suffix(".dwg")).resolve()
    dwg_path.parent.mkdir(parents=True, exist_ok=True)

    preferred = (os.environ.get("DWG_CONVERTER") or "auto").lower()
    engines = detect_engines()

    order: list[str] = []
    if preferred == "libredwg":
        order = ["libredwg", "oda"]
    elif preferred == "oda":
        order = ["oda", "libredwg"]
    else:
        order = ["libredwg", "oda"]

    errors: list[str] = []
    for engine in order:
        if engine == "libredwg" and engines["libredwg"]:
            try:
                return _convert_libredwg(dxf_path, dwg_path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"libredwg: {exc}")
                logger.exception("LibreDWG conversion failed")
        if engine == "oda" and engines["oda"]:
            try:
                return _convert_oda(dxf_path, dwg_path, engines["oda_path"])
            except Exception as exc:  # noqa: BLE001
                errors.append(f"oda: {exc}")
                logger.exception("ODA conversion failed")

    msg = "DWG converter unavailable"
    if errors:
        msg = "; ".join(errors)
    elif not engines["libredwg"] and not engines["oda"]:
        msg = (
            "No DWG engine found. Install LibreDWG (dxf2dwg) or set ODA_FC_PATH "
            "to ODA File Converter."
        )
    return ConvertResult(None, "none", msg)


def _convert_libredwg(dxf_path: Path, dwg_path: Path) -> ConvertResult:
    # LibreDWG refuses overwrite without -y
    if dwg_path.exists():
        dwg_path.unlink()
    cmd = [
        "dxf2dwg",
        "-y",
        "--as",
        "r2000",
        "-o",
        str(dwg_path),
        str(dxf_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    if proc.returncode != 0 or not dwg_path.exists():
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "dxf2dwg failed")
    return ConvertResult(dwg_path, "libredwg", "OK")


def _convert_oda(dxf_path: Path, dwg_path: Path, oda_bin: str) -> ConvertResult:
    """
    ODAFileConverter <in_dir> <out_dir> <out_ver> <out_fmt> <recurse> <audit> [filter]

    out_fmt: DWG, out_ver e.g. ACAD2018
    """
    in_dir = dxf_path.parent
    out_dir = dwg_path.parent / "_oda_out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Work on a copy with simple ASCII name — ODA is picky with filters
    work_dxf = in_dir / "_convert_in.dxf"
    shutil.copy2(dxf_path, work_dxf)
    try:
        cmd = [
            oda_bin,
            str(in_dir),
            str(out_dir),
            "ACAD2018",
            "DWG",
            "0",
            "1",
            "_convert_in.dxf",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=180)
        produced = out_dir / "_convert_in.dwg"
        if not produced.exists():
            # ODA sometimes keeps original stem
            candidates = list(out_dir.glob("*.dwg"))
            if not candidates:
                raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "ODA produced no DWG")
            produced = candidates[0]
        if dwg_path.exists():
            dwg_path.unlink()
        shutil.move(str(produced), str(dwg_path))
    finally:
        work_dxf.unlink(missing_ok=True)
        shutil.rmtree(out_dir, ignore_errors=True)

    return ConvertResult(dwg_path, "oda", "OK")


def build_download_package(
    dxf_path: Path,
    package_path: Path,
    metadata: dict | None = None,
) -> tuple[Path, ConvertResult]:
    """Create ZIP with DXF + DWG (if conversion succeeded) + short readme."""
    metadata = metadata or {}
    conv = convert_dxf_to_dwg(dxf_path, dxf_path.with_suffix(".dwg"))
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(dxf_path, arcname=dxf_path.name)
        if conv.dwg_path and conv.dwg_path.exists():
            zf.write(conv.dwg_path, arcname=conv.dwg_path.name)
        readme = (
            "Image → Drawing package\n"
            f"Object: {metadata.get('object_name', '')}\n"
            f"Code: {metadata.get('code', '')}\n"
            f"DXF: {dxf_path.name}\n"
            f"DWG engine: {conv.engine}\n"
            f"DWG status: {conv.message}\n"
            "Standards: ГОСТ Р 21.101-2020, ГОСТ 2.307-2011, ГОСТ 2.303-68\n"
        )
        zf.writestr("README.txt", readme)
    return package_path, conv
