# syntax=docker/dockerfile:1.7

# ────────────────────────────────────────────────────────────
# Stage 1 — Builder: install dependencies into a virtualenv
# ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create virtualenv; keeps final image clean
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ────────────────────────────────────────────────────────────
# Stage 2 — Runtime: minimal image with app code
# ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    SAGA_CONFIG=/app/config.yaml

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system saga \
    && useradd --system --gid saga --home-dir /app saga

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Application source
COPY --chown=saga:saga saga/ ./saga/
COPY --chown=saga:saga config.example.yaml ./config.example.yaml
COPY --chown=saga:saga pyproject.toml ./pyproject.toml

# Writable runtime dirs (will be overlaid by volumes in production)
RUN mkdir -p /app/db /app/cache/sessions /app/logs/turns \
    && chown -R saga:saga /app

USER saga

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "-m", "saga"]
