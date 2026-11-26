# Service 5 : api-gateway (FastAPI + Uvicorn): the only entry point of the platform (/analyze_pcg)
# Multi-stage build. Build context = the repo root: docker build -f docker/api-gateway.Dockerfile .

# ---------- stage 1 (builder): install the libraries in /install ----------
FROM python:3.12-slim AS builder
WORKDIR /build
COPY api/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- stage 2 (runtime) ----------
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /install /usr/local
# our code and the HTML page
COPY api/main.py api/index.html ./

# the API listens on port 8000
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
