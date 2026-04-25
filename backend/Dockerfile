# ── Stage 1: dependency installer ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /usr/local/bin/uv

# Build-time system deps (compilers needed by some wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a local virtual environment.
# --no-install-project: install only third-party deps; project source is
# copied in the runtime stage so we don't invalidate the cache on every change.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

# Pre-download models at build time using uv (venv has no pip, uv run works).
# spaCy en_core_web_sm (~12 MB) + all-MiniLM-L6-v2 (~90 MB)
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface
RUN uv run python -m spacy download en_core_web_sm \
    && uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Stage 2: runtime image ──────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# libgomp1: required by sentence-transformers (OpenMP)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment and pre-downloaded model cache from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.cache /app/.cache

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Pin HuggingFace cache inside the image so the model is baked in
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    # Limit OpenBLAS threads to prevent memory over-allocation during embedding
    OPENBLAS_NUM_THREADS=2 \
    OPENBLAS_VERBOSE=0 \
    # Disable dynamic thread adjustment to prevent runaway allocation
    VECLIB_MAXIMUM_THREADS=2 \
    MKL_NUM_THREADS=2

# Create non-root user and hand over ownership
RUN useradd --create-home --uid 1000 lexi \
    && chown -R lexi:lexi /app

USER lexi

# Copy application source code
COPY --chown=lexi:lexi src/ ./src/

# Runtime data directories (overridden by Docker volumes in compose)
RUN mkdir -p data judgement_pdfs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
