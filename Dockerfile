# Stage 1: builder
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install-deps -r requirements.txt \
    && rm -rf /install-deps/lib/python3.11/site-packages/torch* \
              /install-deps/lib/python3.11/site-packages/nvidia* \
              /install-deps/lib/python3.11/site-packages/torchvision* \
              /install-deps/lib/python3.11/site-packages/torchaudio* || true

# ============================================================
# Stage 2: runtime — only runtime libs, no compilers
# ============================================================
FROM python:3.11-slim AS runtime

# libgomp1 is the ONLY runtime lib needed (LightGBM/OpenMP), ~10 MB
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install-deps /usr/local

# Copy application code
COPY app/ ./app/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY config.yaml ./

# Logs directory (writable by appuser)
RUN mkdir -p /app/logs

# Mimic the .pth file from com_env
ENV PYTHONPATH=/app/src:/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]