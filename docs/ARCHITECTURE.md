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
   `ezdxf` пишет DXF. DWG-конвертация (LibreDWG/ODA) **убрана** из продукта —
   DXF открывается в AutoCAD / LibreCAD без тяжёлой сборки образа.

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

## Реализовано (v0.3)

- WebGL-просмотр с переключением слоёв
- OpenCV perspective для фото + эвристика окон на фасаде
- DXF через ezdxf (основной формат выдачи)
- ZIP-пакет DXF + README (опционально)
- `AI_MODE=full` интерфейс (YOLO+OCR) с fallback на stub
- PDF → PNG через pypdfium2 (до 3 стр./файл, 150 DPI)
- Stub-геометрия: structural grid + hatch suppression
- Контрольный отрезок (мм) в UI; `needs_scale` по confidence
- Кросс-валидация: голосование модулей + унификация осей + перенос высот
- Штамп Формы 3 (плотная сетка) + portrait fallback

## Паттерны лёгкости / скорости

1. **Stub-first / DXF-only:** без компиляции LibreDWG в Docker.
2. **Downscale-then-scale:** CV на ≤1600/1280 px.
3. **PDF cap:** первые N страниц @ 150 DPI.
4. **Lazy imports:** YOLO/OCR только в нужной стадии.
5. **R2000 DXF:** широкая совместимость CAD-клиентов.

## Следующие шаги (качество, не «вес»)

- ONNX nano denoise / wall-seg вместо полного PyTorch в образе
- ROI-OCR только вокруг размерных цепочек
- Параллель листов через process pool при ≥2 CPU
- Дообучение `walls_seg.pt` на обмерных листах
