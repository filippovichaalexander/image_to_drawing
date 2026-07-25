# Image → Drawing (image_to_drawing)

Веб-приложение для автоматической векторизации архитектурных сканов и фото
в **DXF** по правилам СПДС / ЕСКД (ГОСТ).

## Стек

| Слой | Технология |
|------|------------|
| UI + API | FastAPI + Jinja2 (без React/Node) |
| Просмотр | WebGL (слои, зум, панорама) |
| Очистка сканов | OpenCV; ControlNet — позже |
| Геометрия / OCR | `AI_MODE=stub` или `full` (YOLOv8-seg + PaddleOCR) |
| Фото | OpenCV Perspective Transform |
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

- `AI_MODE=stub|full` — full: YOLO + PaddleOCR (`requirements-full.txt` + `models/walls_seg.pt`)
- `INSTALL_FULL_AI=1` — полный AI-стек при сборке образа

Веса: `python scripts/download_models.py` (или при сборке Docker).

Скачивание: кнопка **Скачать DXF** (`/api/jobs/{id}/download`). Опционально ZIP: `/api/jobs/{id}/package`.

## Документация

- ТЗ: [`main-instruction.md`](./main-instruction.md)
- Архитектура: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)
