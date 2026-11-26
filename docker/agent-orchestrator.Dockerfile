# Service 4 : agent-orchestrator (LangGraph graph of the 3 agents + the RAG + the Pydantic guardrails)
# Multi-stage build. Build context = the repo root: docker build -f docker/agent-orchestrator.Dockerfile .

# ---------- stage 1 (builder): install the libraries of the agents and of the RAG in /install ----------
FROM python:3.12-slim AS builder
WORKDIR /build
COPY agents/requirements.txt agents-requirements.txt
COPY rag_engine/requirements.txt rag-requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r agents-requirements.txt -r rag-requirements.txt

# ---------- stage 2 (runtime) ----------
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /install /usr/local

# the embedding model of the RAG is downloaded ONCE, when the image is built (~220 MB):
# at run time the platform needs no internet (no data leaves the Docker network)
ENV FASTEMBED_CACHE_PATH=/app/fastembed_cache
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# our code: the agents (graph, prompts, schemas, API) and the RAG (ingest, search, evaluation)
COPY agents ./agents
COPY rag_engine ./rag_engine

# the API listens on port 8002
EXPOSE 8002
CMD ["uvicorn", "agents.main:app", "--host", "0.0.0.0", "--port", "8002"]
