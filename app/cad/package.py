"""Package DXF download (no DWG conversion)."""

from __future__ import annotations

import zipfile
from pathlib import Path


def build_download_package(
    dxf_path: Path,
    package_path: Path,
    metadata: dict | None = None,
) -> Path:
    """ZIP with DXF + short README (AutoCAD / LibreCAD open DXF natively)."""
    metadata = metadata or {}
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(dxf_path, arcname=dxf_path.name)
        readme = (
            "Image → Drawing package\n"
            f"Object: {metadata.get('object_name', '')}\n"
            f"Code: {metadata.get('code', '')}\n"
            f"DXF: {dxf_path.name}\n"
            "Format: DXF (open in AutoCAD, LibreCAD, DraftSight, BricsCAD).\n"
            "Standards: ГОСТ Р 21.101-2020, ГОСТ 2.307-2011, ГОСТ 2.303-68\n"
        )
        zf.writestr("README.txt", readme)
    return package_path
