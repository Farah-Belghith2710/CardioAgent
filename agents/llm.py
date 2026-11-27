"""
A thin wrapper around the HTTP API of Ollama (Service 1: llm-server), so the tests can replace it by a fake LLM.

    chat_json(system_prompt, user_prompt) -> {"content": dict, "prompt_tokens": int, "eval_tokens": int, "seconds": float}

format "json": Ollama forces the model to answer with valid JSON. temperature 0.2: stable answers, close to the facts.
Everything stays local (Ollama runs in our Docker network): no external API, no data leaves the platform.
"""
import json
import os

import requests

LLM_URL = os.getenv("LLM_URL", "http://localhost:11434")              # Ollama
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")             # "cardio-llm" once the fine-tuned GGUF is there
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "600"))                  # seconds: an LLM on a CPU can be slow


def chat_json(system_prompt, user_prompt):
    response = requests.post(
        LLM_URL + "/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 4096},
        },
        timeout=LLM_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "content": json.loads(data["message"]["content"]),     # a ValueError if the text is not valid JSON
        "prompt_tokens": data.get("prompt_eval_count", 0),     # tokens read by the model
        "eval_tokens": data.get("eval_count", 0),              # tokens written by the model
        "seconds": data.get("total_duration", 0) / 1e9,        # Ollama gives nanoseconds
    }


def is_ready():
    # is Ollama running, and does it have our model?
    try:
        tags = requests.get(LLM_URL + "/api/tags", timeout=5).json()
    except requests.RequestException:
        return False
    for model in tags.get("models", []):
        if model["name"] == LLM_MODEL or model["name"] == LLM_MODEL + ":latest":
            return True
    return False
