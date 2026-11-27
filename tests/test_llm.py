"""
Tests of agents/llm.py in the online mode (Groq / xAI, OpenAI-style API) with FAKE HTTP answers (no key, no network):
the provider guessed from the key, the request, the JSON answer, the retry, the errors, and the key never shown.
The local mode (Ollama) is tested in test_agents.py.
"""
import sys
import types

import pytest
import requests
from fastapi.testclient import TestClient

import agents.main as orchestrator
import api.main as gateway
from agents import chat, graph, llm

KEY = "gsk_TEST_secret_key_123456"


class FakeResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


def answer_of(content, prompt_tokens=50, completion_tokens=20):
    # what Groq / xAI send back
    return FakeResponse(200, {"choices": [{"message": {"content": content}}],
                              "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}})


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setattr(llm, "LLM_API_KEY", KEY)
    monkeypatch.setattr(llm, "LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)    # no real waiting before the retry


def test_provider_is_guessed_from_the_key(monkeypatch):
    assert llm.provider() == "ollama"                                     # no key: the local mode
    monkeypatch.setattr(llm, "LLM_API_KEY", "gsk_abc")
    assert llm.provider() == "groq" and llm.base_url() == "https://api.groq.com/openai/v1"
    monkeypatch.setattr(llm, "LLM_API_KEY", "xai-abc")
    assert llm.provider() == "xai" and llm.base_url() == "https://api.x.ai/v1"
    monkeypatch.setattr(llm, "LLM_PROVIDER", "ollama")                    # the switch of the local mode wins
    assert llm.provider() == "ollama" and llm.is_online() is False
    monkeypatch.setattr(llm, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(llm, "LLM_BASE_URL", "https://example.test/v1/")
    assert llm.base_url() == "https://example.test/v1"


def test_the_request_is_openai_style_and_the_json_is_read(monkeypatch, groq):
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return answer_of('{"answer": "bonjour"}')

    monkeypatch.setattr(llm.requests, "post", fake_post)
    result = llm.chat_json("system prompt", "user prompt", max_tokens=300)
    assert sent["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert sent["headers"] == {"Authorization": "Bearer " + KEY} and sent["timeout"] == 60
    body = sent["json"]
    assert body["model"] == "llama-3.3-70b-versatile" and body["temperature"] == 0.2 and body["max_tokens"] == 300
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [{"role": "system", "content": "system prompt"}, {"role": "user", "content": "user prompt"}]
    assert result["content"] == {"answer": "bonjour"} and result["prompt_tokens"] == 50 and result["eval_tokens"] == 20
    assert result["seconds"] >= 0


def test_one_retry_after_429(monkeypatch, groq):
    answers = [FakeResponse(429, {"error": {"message": "rate limit"}}), answer_of('{"ok": true}')]
    monkeypatch.setattr(llm.requests, "post", lambda url, **kwargs: answers.pop(0))
    assert llm.chat_json("s", "u")["content"] == {"ok": True} and answers == []


def test_second_error_is_not_retried_again(monkeypatch, groq):
    calls = []

    def always_503(url, **kwargs):
        calls.append(url)
        return FakeResponse(503, {})

    monkeypatch.setattr(llm.requests, "post", always_503)
    with pytest.raises(llm.LLMUnavailable):
        llm.chat_json("s", "u")
    assert len(calls) == 2


def test_refused_key_gives_a_clear_error_without_the_key(monkeypatch, groq, capsys):
    monkeypatch.setattr(llm.requests, "post", lambda url, **kwargs: FakeResponse(401, {"error": {"message": "Invalid API Key"}}))
    with pytest.raises(llm.LLMUnavailable) as error:
        llm.chat_json("s", "u")
    printed = capsys.readouterr().out
    assert "refused" in str(error.value) and "401" in printed
    assert KEY not in str(error.value) and KEY not in printed


def test_no_network_gives_a_clear_error_without_the_key(monkeypatch, groq, capsys):
    def no_network(url, **kwargs):
        raise requests.ConnectionError("Max retries exceeded with url: /openai/v1/chat/completions")

    monkeypatch.setattr(llm.requests, "post", no_network)
    with pytest.raises(llm.LLMUnavailable) as error:
        llm.chat_json("s", "u")
    assert KEY not in str(error.value) and KEY not in capsys.readouterr().out


def test_no_key_means_no_request(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "groq")

    def must_not_be_called(url, **kwargs):
        raise AssertionError("no request without a key")

    monkeypatch.setattr(llm.requests, "post", must_not_be_called)
    with pytest.raises(llm.LLMUnavailable):
        llm.chat_json("s", "u")


def test_the_default_model_comes_from_the_list_of_the_account(monkeypatch):
    monkeypatch.setattr(llm, "LLM_API_KEY", KEY)
    asked = []

    def fake_get(url, headers=None, timeout=None):
        asked.append((url, headers))
        return FakeResponse(200, {"data": [{"id": "whisper-large-v3"}, {"id": "llama-3.1-8b-instant"}]})

    monkeypatch.setattr(llm.requests, "get", fake_get)
    assert llm.model_name() == "llama-3.1-8b-instant" and llm.is_ready() is True
    assert asked == [("https://api.groq.com/openai/v1/models", {"Authorization": "Bearer " + KEY})]    # once


def test_is_ready_is_false_with_a_refused_key(monkeypatch):
    monkeypatch.setattr(llm, "LLM_API_KEY", KEY)
    monkeypatch.setattr(llm.requests, "get", lambda url, **kwargs: FakeResponse(401, {}))
    assert llm.is_ready() is False
    assert llm.model_name() == "llama-3.3-70b-versatile"          # the first favourite, if the list is unknown


# ==================== what the user sees when the online LLM is down ====================
def test_chat_says_the_online_service_is_down(monkeypatch, audio_result):
    monkeypatch.setattr(chat, "safe_search", lambda question, k=4, category=None: [])

    def refused(system_prompt, user_prompt, max_tokens=None):
        raise llm.LLMUnavailable("Groq: the API key was refused (401), check LLM_API_KEY in .env")

    monkeypatch.setattr(llm, "chat_json", refused)
    answer, metrics = chat.chat("Dois-je voir un médecin ?", {"audio": audio_result})
    assert answer["answer"].startswith(chat.ONLINE_DOWN["fr"]) and metrics["safe_answer"] is True
    assert "cardiologue" in answer["answer"]                       # abnormal result: always the advice of a doctor


def test_report_falls_back_without_the_key(monkeypatch, audio_result):
    monkeypatch.setattr(graph, "rag_search", lambda question, k=4, category=None: [])
    monkeypatch.setattr(llm, "LLM_API_KEY", KEY)
    monkeypatch.setattr(llm, "LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setattr(llm.requests, "post", lambda url, **kwargs: FakeResponse(401, {}))
    report, metrics = graph.run_agents(audio_result)
    assert metrics["fallback"] is True and report["audit"]["approved"] is False
    assert KEY not in str(report) and KEY not in str(metrics)


def test_the_file_name_is_never_sent_to_the_online_llm(monkeypatch, groq, audio_result):
    prompts_sent = []

    def fake_post(url, headers=None, json=None, timeout=None):
        prompts_sent.append(str(json["messages"]))
        return answer_of('{"answer": "Le son est classé anormal. Parlez-en à un médecin.", "sources_used": []}')

    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(chat, "safe_search", lambda question, k=4, category=None: [])
    chat.chat("Ce son est-il normal ?", {"audio": audio_result})
    assert prompts_sent and all("patient_042" not in text for text in prompts_sent)


# ==================== health and MLflow ====================
def test_health_gives_the_provider_and_never_the_key(monkeypatch, groq):
    monkeypatch.setattr(orchestrator.requests, "get", lambda url, **kwargs: FakeResponse(200, {"data": []}))
    data = TestClient(orchestrator.app).get("/health").json()
    assert data["llm_provider"] == "groq" and data["llm_online"] is True
    assert data["llm_model"] == "llama-3.3-70b-versatile" and KEY not in str(data)


def test_gateway_health_gives_the_mode(monkeypatch):
    monkeypatch.setattr(gateway, "ORCHESTRATOR_URL", "http://agent-orchestrator:8002/orchestrate")
    monkeypatch.setattr(gateway.requests, "get", lambda url, timeout=None: FakeResponse(200, {"llm_provider": "ollama"}))
    assert TestClient(gateway.app).get("/health").json()["llm_provider"] == "ollama"

    def down(url, timeout=None):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(gateway.requests, "get", down)
    assert TestClient(gateway.app).get("/health").json()["llm_provider"] is None


def test_mlflow_gets_the_provider_and_the_model_never_the_key(monkeypatch, groq):
    logged = {}

    class FakeClient:
        def __init__(self, uri):
            pass

        def get_experiment_by_name(self, name):
            return None

        def create_experiment(self, name):
            return "1"

        def create_run(self, experiment_id, run_name=None):
            return types.SimpleNamespace(info=types.SimpleNamespace(run_id="run1"))

        def log_param(self, run_id, name, value):
            logged[name] = value

        def log_metric(self, run_id, name, value):
            logged[name] = value

        def set_terminated(self, run_id):
            pass

    fake_tracking = types.ModuleType("mlflow.tracking")
    fake_tracking.MlflowClient = FakeClient
    monkeypatch.setitem(sys.modules, "mlflow", types.ModuleType("mlflow"))
    monkeypatch.setitem(sys.modules, "mlflow.tracking", fake_tracking)
    monkeypatch.setattr(orchestrator, "MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    orchestrator.log_chat({"total_s": 1.2, "on_topic": True, "guardrail": "keyword"})
    assert logged["llm_provider"] == "groq" and logged["llm_model"] == "llama-3.3-70b-versatile"
    assert KEY not in str(logged)
