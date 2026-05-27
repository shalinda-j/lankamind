# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy dependency files first (layer-cache friendly)
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies into /install
RUN pip install --no-cache-dir --prefix=/install \
    torch==2.2.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install \
    transformers>=4.35.0 \
    accelerate>=0.24.0 \
    "pyzmq>=25.0.0" \
    click>=8.1.0 \
    fastapi>=0.100.0 \
    "uvicorn[standard]>=0.23.0" \
    httpx>=0.24.0 \
    "pydantic>=2.0.0" \
    tokenizers>=0.14.0

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY . .

# Create data directory for ledger + keys
RUN mkdir -p /root/.lankamind

# Default environment
ENV PYTHONPATH=/app
ENV LANKAMIND_BASE_PORT=5500
ENV LANKAMIND_RESULT_PORT=5599

EXPOSE 8000 5500 5501 5502 5599 5700 5701 9090 6000

# Default command: show help
ENTRYPOINT ["python", "-m", "cli.main"]
CMD ["--help"]
