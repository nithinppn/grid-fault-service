# ---- build stage: train the model (needs the full training deps) ----
FROM python:3.12-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt requirements-train.txt ./
RUN pip install -r requirements-train.txt
COPY src ./src
# the model is trained here, not committed to git: reproducible and matches this image's library versions
RUN PYTHONPATH=src python -m gridfault.train --no-tracking

# ---- runtime stage: slim image with only serving deps ----
FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY src ./src
COPY --from=builder /app/artifacts ./artifacts
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"
CMD ["uvicorn", "gridfault.api:app", "--host", "0.0.0.0", "--port", "8000"]
