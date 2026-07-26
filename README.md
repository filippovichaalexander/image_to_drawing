# Image → Drawing (image_to_drawing)

Веб-приложение для автоматической векторизации архитектурных сканов и фото
в **DXF** по правилам СПДС / ЕСКД (ГОСТ).

## Стек

| Слой | Технология |
|------|------------|
| UI + API | FastAPI + Jinja2 (без React/Node) |
| Просмотр | WebGL (слои, зум, панорама) |
| Очистка сканов | OpenCV; ControlNet — позже |
| Геометрия / OCR | `AI_MODE=stub` или `full` (YOLOv8-seg Model 1 + PaddleOCR) |
| Фото | OpenCV perspective + **ORB** photo↔plan (C3Po optional later) |
| CAD | ezdxf → **DXF** (слои, размеры, рамка/штамп ГОСТ) |
| PDF | pypdfium2 → PNG |
| Runtime | Docker Compose (без LibreDWG) |

## Быстрый старт

```bash
docker compose up --build
```

Откройте http://localhost:8000 · Health: `/health` · Capabilities: `/api/capabilities`

## Переменные окружения

См. `.env.example`.

- `AI_MODE=stub|full` — full: YOLO + PaddleOCR (`requirements-full.txt` + `models/floorplan-seg.pt`)
- `INSTALL_FULL_AI=1` — полный AI-стек при сборке образа

Веса:
```bash
python scripts/download_models.py          # Model 1 floorplan-seg
# Photo alignment uses OpenCV ORB by default (no download).
# Optional later: python scripts/download_models.py --c3 --confirm-large  # ~10 GB C3Po
```

Скачивание: кнопка **Скачать DXF** (`/api/jobs/{id}/download`). Опционально ZIP: `/api/jobs/{id}/package`.

## Документация

- ТЗ: [`main-instruction.md`](./main-instruction.md)
- Архитектура: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)
- Future TODO (C3Po upgrade): [`docs/TODO.md`](./docs/TODO.md)
