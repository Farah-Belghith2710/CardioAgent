#!/bin/sh
# Service 1 (llm-server): start Ollama, then make sure that the model used by the agents ($LLM_MODEL) is there.
#  - the fine-tuned GGUF exists in /models (= llm_server/models/ on the PC) -> "ollama create cardio-llm"
#  - the model is still missing -> "ollama pull $LLM_MODEL" (the base model, downloaded once, kept in the volume)

ollama serve &
SERVER=$!

# wait until the Ollama server answers
until ollama list > /dev/null 2>&1; do
    sleep 1
done

GGUF=/models/cardio-qwen2.5-7b-q4_k_m.gguf
if [ -f "$GGUF" ]; then
    echo "Fine-tuned GGUF found: creating the model cardio-llm"
    # the Modelfile says FROM ./cardio-...gguf: here the GGUF is in /models
    sed "s#^FROM .*#FROM $GGUF#" /app/Modelfile > /tmp/Modelfile
    ollama create cardio-llm -f /tmp/Modelfile
fi

if ! ollama show "$LLM_MODEL" > /dev/null 2>&1; then
    echo "Downloading the model $LLM_MODEL (only the first time)"
    ollama pull "$LLM_MODEL"
fi
echo "llm-server ready with the model $LLM_MODEL"

# keep the container running as long as the Ollama server runs
wait $SERVER
