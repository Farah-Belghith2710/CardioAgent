# Service 2 : vector-db (Qdrant, official image): the vectors (embeddings) of the RAG documents, cosine search
# The collection "cardio_guidelines" is filled by: docker compose run --rm agent-orchestrator python -m rag_engine.ingest
# The data is kept in the volume qdrant-data (see docker-compose.yml).
FROM qdrant/qdrant:v1.20.2

# REST API on port 6333 (gRPC on 6334)
EXPOSE 6333
