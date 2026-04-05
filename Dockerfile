# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py tasks.py grader.py env.py server.py inference.py openenv.yaml ./

# Set correct permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# ─── Runtime config ───────────────────────────────────────────────────────────

# HuggingFace Spaces expects port 7860
EXPOSE 7860

# Environment variables with defaults (override at runtime)
ENV API_BASE_URL=http://localhost:7860
ENV MODEL_NAME=gpt-4o-mini
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
