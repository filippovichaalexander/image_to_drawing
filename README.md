# Image → Drawing (image_to_drawing)

Веб-приложение для автоматической векторизации архитектурных сканов и фото
в DXF/DWG по правилам СПДС / ЕСКД (ГОСТ).

## Стек

| Слой | Технология |
|------|------------|
| UI + API | FastAPI + Jinja2 (без React/Node) |
| Просмотр | WebGL (слои, зум, панорама) |
| Очистка сканов | OpenCV; ControlNet — позже |
| Геометрия / OCR | `AI_MODE=stub` (контуры) или `full` (YOLOv8-seg + PaddleOCR) |
| Фото | OpenCV Perspective Transform |
| CAD | ezdxf (слои, размеры, рамка/штамп ГОСТ) |
| DWG | LibreDWG `dxf2dwg` (в образе) или ODA File Converter (`ODA_FC_PATH`) |
| Runtime | Docker Compose |

## Быстрый старт

```bash
docker compose up --build
```

Откройте http://localhost:8000 · Health: `/health` · Capabilities: `/api/capabilities`

## Переменные окружения

См. `.env.example`.

- `AI_MODE=stub|full` — full требует `requirements-full.txt` и веса в `models/`
- `DWG_CONVERTER=auto|libredwg|oda`
- `ODA_FC_PATH` — путь к ODA File Converter (опционально)

Скачивание: ZIP с DXF + DWG (если конвертер отработал) + README.

## Документация

- ТЗ: [`main-instruction.md`](./main-instruction.md)
- Архитектура / коллизии: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)
