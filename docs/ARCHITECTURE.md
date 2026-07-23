# Analysis of TZ collisions & optimization notes
# (tracked for project decisions; Russian)

## Коллизии ТЗ

1. **Легковесность vs тяжёлый ИИ-стек**
   ControlNet + SAM + YOLOv8 + PaddleOCR + PyTorch + Meshroom противоречат
   «легковесному» Docker-образу и быстрому cold start. Решение: `AI_MODE=stub|full`,
   MVP на OpenCV; тяжёлые модели — отдельный профиль / GPU-воркер.

2. **15–20 с на пакет (3 чертежа + 2 фото)**
   Полный пайплайн с ControlNet на CPU нереалистичен. Нужен GPU и/или
   замена denoising на классические фильтры OpenCV + лёгкие ONNX-модели.

3. **DWG vs DXF**
   `ezdxf` пишет DXF. Настоящий DWG — через ODA File Converter / LibreDWG
   (отдельный контейнер). В UI: «пакет DXF», конвертация DWG — опция.

4. **FastHTML «без Node» vs WebGL-просмотрщик**
   PixiJS / dxf-viewer — JS с CDN. Node-сборка не нужна; конфликт мнимый.

5. **Meshroom API**
   Тяжёлый, не Python-native, плохо стыкуется с 15–20 с SLA.
   MVP: OpenCV Perspective Transform; Meshroom — post-MVP.

6. **Публичные облачные AI API**
   ТЗ описывает локальный Python CV-стек, не SaaS.
   Публичные API (OpenAI Vision и т.п.) **не обязательны** и ухудшают
   предсказуемость ГОСТ-конвейера / privacy сканов. Допустимы только как
   опциональный эксперимент вне основного пайплайна.

## Оптимизации на перспективу

- Бинаризация/очистка: OpenCV → опционально лёгкий ONNX denoise
- Сегментация стен: YOLOv8n-seg или MobileSAM вместо тяжёлого SAM
- OCR: PaddleOCR mobile / только ROI вокруг размерных линий
- Кэш моделей в volume; warm worker
- Параллельная обработка листов (asyncio + process pool)
- CDN для dxf-viewer; критический CSS inline

## Реализовано (v0.2+)

- WebGL-просмотр с переключением слоёв
- OpenCV perspective для фото
- DXF→DWG через LibreDWG (`dxf2dwg`) + fallback ODA
- ZIP-пакет DWG/DXF
- `AI_MODE=full` интерфейс (YOLO+OCR) с fallback на stub
- PDF → PNG через pypdfium2 (до 3 стр./файл, 150 DPI)

## Паттерны лёгкости / скорости

1. **Stub-first:** `AI_MODE=stub` по умолчанию; тяжёлые веса только в `models/` + `requirements-full.txt`.
2. **Downscale-then-scale:** CV на ≤1600 px по длинной стороне, координаты в мм через scale factor.
3. **PDF cap:** первые N страниц @ 150 DPI — хватает для линий, без печати.
4. **Lazy imports:** YOLO/OCR/pypdfium тянутся только в нужной стадии.
5. **R2000 DXF:** совместимость с LibreDWG без тяжёлого ODA в образе.
