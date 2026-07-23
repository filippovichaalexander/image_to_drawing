"""Lightweight PDF → PNG (first pages only) for the denoise stage."""

from __future__ import annotations

from pathlib import Path

# Cap pages so a 40-page scan set does not blow CPU/RAM on the MVP path.
MAX_PAGES = 3
# Render DPI: enough for line work, far below print DPI (fast + light).
RENDER_DPI = 150


def rasterize_pdf(src: Path, dest_dir: Path, max_pages: int = MAX_PAGES) -> list[Path]:
    """
    Render up to `max_pages` of a PDF into PNG files.

    Uses pypdfium2 (PDFium wheel) — no poppler apt package, keeps the slim image.
    """
    import pypdfium2 as pdfium

    dest_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(src))
    try:
        n = min(len(pdf), max_pages)
        scale = RENDER_DPI / 72.0
        out: list[Path] = []
        for i in range(n):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil()
            path = dest_dir / f"{src.stem}_p{i + 1:02d}.png"
            pil.save(path, format="PNG", optimize=True)
            out.append(path)
        return out
    finally:
        pdf.close()


def expand_drawing_inputs(paths: list[Path], work_dir: Path) -> list[Path]:
    """Turn mixed image/PDF uploads into a flat list of raster paths for Stage 1."""
    raster_dir = work_dir / "pdf_raster"
    expanded: list[Path] = []
    for path in paths:
        if path.suffix.lower() == ".pdf":
            expanded.extend(rasterize_pdf(path, raster_dir))
        else:
            expanded.append(path)
    return expanded
