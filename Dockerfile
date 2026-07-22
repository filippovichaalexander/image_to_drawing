# ECR public mirror of library/python — often more reachable than docker.io
FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm AS libredwg-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        curl \
        python3 \
    && rm -rf /var/lib/apt/lists/*

# LibreDWG provides dxf2dwg (open-source DXF→DWG). Pin a known release.
ARG LIBREDWG_VERSION=0.13.3
WORKDIR /tmp
RUN curl -fsSL "https://ftp.gnu.org/gnu/libredwg/libredwg-${LIBREDWG_VERSION}.tar.gz" \
      -o libredwg.tar.gz \
    && tar -xzf libredwg.tar.gz \
    && cd "libredwg-${LIBREDWG_VERSION}" \
    && ./configure --prefix=/opt/libredwg --disable-python --disable-shared \
    && make -j"$(nproc)" -C src \
    && make -j"$(nproc)" -C programs dxf2dwg \
    && make -C src install \
    && make -C programs install-exec \
    && strip /opt/libredwg/bin/dxf2dwg || true \
    && rm -rf /tmp/libredwg* \
    && /opt/libredwg/bin/dxf2dwg --version

FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app \
    PORT=8000 \
    PATH="/opt/libredwg/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/libredwg/lib:${LD_LIBRARY_PATH}" \
    DWG_CONVERTER=auto

WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=libredwg-build /opt/libredwg /opt/libredwg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

RUN mkdir -p /app/data/uploads /app/data/results \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app \
    && chmod -R a+rwX /app/data

# Bind mounts often use host UID; default image user is non-root.
# Compose overrides with HOST_UID/HOST_GID when needed.
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
