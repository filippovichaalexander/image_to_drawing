# Slim runtime — DXF via ezdxf; no LibreDWG compile (DWG removed)
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app \
    PORT=8000 \
    AI_MODE=full \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
        libgfortran5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-full.txt ./

RUN pip install --no-cache-dir -r requirements.txt

ARG INSTALL_FULL_AI=1
RUN if [ "$INSTALL_FULL_AI" = "1" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install --no-cache-dir -r requirements-full.txt \
      && python -c "import torch, ultralytics, paddleocr; print('AI ok', torch.__version__)"; \
    fi

COPY app ./app
COPY scripts ./scripts

RUN mkdir -p /app/models /app/.ultralytics \
    && if [ "$INSTALL_FULL_AI" = "1" ]; then \
         python scripts/download_models.py \
         && test -f /app/models/walls_seg.pt \
         && ls -lh /app/models/walls_seg.pt; \
       fi

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/uploads /app/data/results \
    && chown -R appuser:appuser /app \
    && chmod -R a+rwX /app/data /app/models /app/.ultralytics

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
