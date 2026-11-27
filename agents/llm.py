"""
The LLM of the agents and of the chat, behind 1 function, so the rest of the code (and the tests) never change:

    chat_json(system_prompt, user_prompt, max_tokens=None) -> {"content": dict, "prompt_tokens": int, "eval_tokens": int, "seconds": float}

2 ways to run it (the settings LLM_* of the file .env, next to docker-compose.yml):
  - online, with an API key (fast: a few seconds), OpenAI-style API "POST <base URL>/chat/completions":
        "groq" = Groq (keys "gsk_..."), "xai" = xAI Grok (keys "xai-...")
  - "ollama" = the local LLM of the container llm-server (the version of the PDF: nothing leaves the PC, slow on a CPU)
  LLM_PROVIDER empty -> guessed from the key (gsk_ -> groq, xai- -> xai); no key -> ollama.
The key is a secret: it only goes in the "Authorization" header of the requests to the provider. It is never printed,
never put in an error message, never logged in MLflow and never sent to the browser.
JSON mode: the model must answer with valid JSON. temperature 0.2: stable answers, close to the facts.
"""
import json
import os
import time

import requests

# the settings (docker-compose.yml gives them to the container agent-orchestrator, from the file .env)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()          # "groq", "xai", "ollama" or "" (guessed)
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()                    # secret
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()                  # "" = the default URL of the provider
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()                        # "" = the default model of the provider
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434")              # Ollama (local mode)
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "600"))                  # seconds, local mode: an LLM on a CPU is slow
ONLINE_TIMEOUT = 60                                                    # seconds, online mode

DEFAULT_BASE_URLS = {"groq": "https://api.groq.com/openai/v1", "xai": "https://api.x.ai/v1"}
OLLAMA_DEFAULT_MODEL = "qwen2.5:3b-instruct"                           # "cardio-llm" = our fine-tuned model
# fast models with a JSON mode that answer well in French, the best first: the first one the account has is used
PREFERRED_MODELS = {
    # groq: tested on Farah's account (27/11/2026): gpt-oss-120b answers in ~0.8 s, valid JSON, good French
    "groq": ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "qwen/qwen3.8-27b", "qwen/qwen3-32b",
             "llama-3.1-8b-instant", "openai/gpt-oss-20b"],
    "xai": ["grok-4-fast-non-reasoning", "grok-3-mini", "grok-3-mini-fast", "grok-4-fast", "grok-4"],
}
NOT_CHAT_MODELS = ["whisper", "tts", "orpheus", "guard", "embed", "image", "vision", "audio"]    # in the list, but not for a chat
NAMES = {"groq": "Groq", "xai": "xAI", "ollama": "Ollama"}


class LLMUnavailable(Exception):
    # the online LLM cannot answer: no key, key refused, no network, too many requests... (the key is never inside)
    pass


# the answer of "GET /models" (online mode), checked at most once a minute: is the key OK, which model to use
_online = {"checked_at": 0.0, "ready": False, "model": None}


def provider():
    if LLM_PROVIDER in ["groq", "xai", "ollama"]:
        return LLM_PROVIDER
    if LLM_API_KEY.startswith("gsk_"):
        return "groq"
    if LLM_API_KEY.startswith("xai-"):
        return "xai"
    return "ollama"


def is_online():
    return provider() in ["groq", "xai"]


def base_url():
    if LLM_BASE_URL != "":
        return LLM_BASE_URL.rstrip("/")
    return DEFAULT_BASE_URLS.get(provider(), "")


def auth_header():
    return {"Authorization": "Bearer " + LLM_API_KEY}


def hide_key(text):
    # safety: if a text ever contained the key, it is replaced by *** before being printed
    if LLM_API_KEY != "":
        text = text.replace(LLM_API_KEY, "***")
    return text


def fail(message):
    # a clear message in the logs of the orchestrator (without the key), then the agents use their safe answers
    message = hide_key(message)
    print("LLM error:", message)
    raise LLMUnavailable(message)


def check_online():
    # GET <base URL>/models with the key (at most once a minute): is the key accepted, and which models exist?
    if time.time() - _online["checked_at"] < 60:
        return
    _online["checked_at"] = time.time()
    _online["ready"] = False
    if LLM_API_KEY == "":
        print("LLM error: no API key (write LLM_API_KEY in the file .env)")
        return
    try:
        response = requests.get(base_url() + "/models", headers=auth_header(), timeout=5)
    except requests.RequestException as error:
        print("LLM error:", NAMES[provider()], "does not answer (" + type(error).__name__ + ")")
        return
    if response.status_code != 200:
        print("LLM error:", NAMES[provider()], "GET /models ->", response.status_code, "(check LLM_API_KEY in .env)")
        return
    _online["ready"] = True
    available = []
    for model in response.json().get("data", []):
        available.append(str(model.get("id", "")))
    for name in PREFERRED_MODELS[provider()]:
        if name in available:
            _online["model"] = name
            return
    for name in available:    # none of our favourites: the first model made for chatting
        if name != "" and not any(word in name.lower() for word in NOT_CHAT_MODELS):
            _online["model"] = name
            return


def model_name():
    # the model used: LLM_MODEL if given, else the default of the provider
    if LLM_MODEL != "":
        return LLM_MODEL
    if is_online():
        check_online()
        return _online["model"] or PREFERRED_MODELS[provider()][0]
    return OLLAMA_DEFAULT_MODEL


def chat_json(system_prompt, user_prompt, max_tokens=None):
    if is_online():
        return chat_json_online(system_prompt, user_prompt, max_tokens)
    return chat_json_ollama(system_prompt, user_prompt, max_tokens)


def chat_json_online(system_prompt, user_prompt, max_tokens=None):
    # Groq or xAI: the same OpenAI-style API
    name = NAMES[provider()]
    if LLM_API_KEY == "":
        fail(name + ": no API key (write LLM_API_KEY in the file .env)")
    body = {
        "model": model_name(),
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "temperature": 0.2,
        # at least 1024: some online models "think" before answering and the thinking counts in max_tokens
        # (the small limit of the chat, 300, is only for the slow local CPU)
        "max_tokens": max(max_tokens or 0, 1024),
        "response_format": {"type": "json_object"},     # JSON mode: the answer is valid JSON
    }
    start = time.perf_counter()
    for attempt in [1, 2]:
        try:
            response = requests.post(base_url() + "/chat/completions", headers=auth_header(), json=body,
                                     timeout=ONLINE_TIMEOUT)
        except requests.RequestException as error:
            fail(name + " does not answer: no network or too slow (" + type(error).__name__ + ")")
        if attempt == 1 and (response.status_code == 429 or response.status_code >= 500):
            time.sleep(2)    # too many requests, or a problem on the provider's side: 1 more try after 2 s
            continue
        break
    if response.status_code in [401, 403]:
        fail(name + ": the API key was refused (" + str(response.status_code) + "), check LLM_API_KEY in .env")
    if response.status_code != 200:
        detail = ""
        try:
            detail = str(response.json()["error"]["message"])[:200]    # e.g. "model not found" (never the key)
        except Exception:
            pass
        fail(name + ": error " + str(response.status_code) + " " + detail)
    data = response.json()
    usage = data.get("usage") or {}
    return {
        "content": json.loads(data["choices"][0]["message"]["content"]),    # a ValueError if it is not valid JSON
        "prompt_tokens": usage.get("prompt_tokens", 0),                     # tokens read by the model
        "eval_tokens": usage.get("completion_tokens", 0),                   # tokens written by the model
        "seconds": round(time.perf_counter() - start, 3),
    }


def chat_json_ollama(system_prompt, user_prompt, max_tokens=None):
    # the local mode (the version of the PDF): Ollama in the container llm-server
    options = {"temperature": 0.2, "num_ctx": 4096}
    if max_tokens is not None:
        options["num_predict"] = max_tokens    # a short answer: the LLM runs on a CPU (the chat uses ~300 tokens)
    response = requests.post(
        LLM_URL + "/api/chat",
        json={
            "model": model_name(),
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "format": "json",
            "stream": False,
            "options": options,
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
    # online: the key is set and the provider answers GET /models | local: Ollama runs and has our model
    if is_online():
        check_online()
        return _online["ready"]
    try:
        tags = requests.get(LLM_URL + "/api/tags", timeout=5).json()
    except requests.RequestException:
        return False
    for model in tags.get("models", []):
        if model["name"] == model_name() or model["name"] == model_name() + ":latest":
            return True
    return False
