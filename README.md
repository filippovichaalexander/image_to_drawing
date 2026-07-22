# Image → Drawing (image_to_drawing)

Веб-приложение для автоматической векторизации архитектурных сканов и фото
в DXF/DWG по правилам СПДС / ЕСКД (ГОСТ).

## Стек (MVP)

| Слой | Технология |
|------|------------|
| UI + API | FastAPI + Jinja2 (лёгкий HTML/CSS/JS, без React/Node) |
| Просмотр | CDN dxf-viewer / WebGL (этап UI) |
| Очистка сканов | OpenCV (лёгкий путь); ControlNet — опционально позже |
| Геометрия / OCR | stubs → YOLOv8-seg + PaddleOCR (профиль `full`) |
| CAD | ezdxf (слои, размеры, рамка и штамп ГОСТ) |
| Runtime | Docker + docker compose |

## Быстрый старт

```bash
docker compose up --build
```

Откройте http://localhost:8000

Healthcheck: http://localhost:8000/health

## Переменные окружения

См. `.env.example`. По умолчанию `AI_MODE=stub` — без тяжёлых моделей
(быстрый старт, малый образ).

## Документация ТЗ

Исходное ТЗ: [`main-instruction.md`](./main-instruction.md)

## Этапы MVP

1. Docker + каркас приложения
2. UI загрузки (чертежи / фото) + прогресс
3. OpenCV denoising
4. ezdxf: слои + рамка/штамп ГОСТ
5. Оркестрация пайплайна + экспорт DXF
