# Service 1 : llm-server (Ollama, official image): runs the quantized LLM (GGUF 4-bit) locally, no external API
# Build context = the repo root: docker build -f docker/llm-server.Dockerfile .
#   - the fine-tuned GGUF (llm_server/models/, mounted in /models) -> the model "cardio-llm" (llm_finetuning/Modelfile)
#   - without it -> the base model $LLM_MODEL is downloaded once (default: qwen2.5:3b-instruct, see docker-compose.yml)
FROM ollama/ollama:0.40.18

# the recipe of our fine-tuned model and the start script
COPY llm_finetuning/Modelfile /app/Modelfile
COPY docker/llm-server-entrypoint.sh /app/entrypoint.sh

# the Ollama API listens on port 11434
EXPOSE 11434
ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
