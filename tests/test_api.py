"""
Integration tests of the API gateway (Service 5) with HTTPX (FastAPI TestClient): the audio engine and the
orchestrator are replaced by fakes, so we test what the gateway does with their answers (400, 503, forwarding).
"""
import requests
from fastapi.testclient import TestClient

import api.main as gateway

client = TestClient(gateway.app)


class FakeResponse:
    # a fake answer of another service (like requests.Response)
    def __init__(self, status_code, data):
        self.status_code = status_code
        self.data = data
        self.text = str(data)

    def json(self):
        return self.data


def test_home_page_is_the_html_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "CardioAgent-Ops" in response.text


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_refuses_a_file_that_is_not_a_wav(heart_wav):
    response = client.post("/analyze_pcg", files={"file": ("heart.mp3", heart_wav, "audio/mpeg")})
    assert response.status_code == 400
    assert ".wav" in response.json()["detail"]


def test_without_audio_engine_gives_the_duration(monkeypatch, heart_wav):
    monkeypatch.setattr(gateway, "AUDIO_ENGINE_URL", "")
    data = client.post("/analyze_pcg", files={"file": ("heart.wav", heart_wav, "audio/wav")}).json()
    assert data["duration_s"] == 6.0
    assert data["size_bytes"] == len(heart_wav)


def test_audio_engine_down_gives_503(monkeypatch, heart_wav):
    monkeypatch.setattr(gateway, "AUDIO_ENGINE_URL", "http://audio-engine:8001/predict")

    def engine_down(url, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(gateway.requests, "post", engine_down)
    response = client.post("/analyze_pcg", files={"file": ("heart.wav", heart_wav, "audio/wav")})
    assert response.status_code == 503
    assert "injoignable" in response.json()["detail"]


def test_error_of_the_audio_engine_is_given_to_the_client(monkeypatch, silent_wav):
    monkeypatch.setattr(gateway, "AUDIO_ENGINE_URL", "http://audio-engine:8001/predict")
    monkeypatch.setattr(gateway.requests, "post",
                        lambda url, **kwargs: FakeResponse(400, {"detail": "Enregistrement silencieux"}))
    response = client.post("/analyze_pcg", files={"file": ("silent.wav", silent_wav, "audio/wav")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Enregistrement silencieux"


def test_audio_result_only_when_no_orchestrator(monkeypatch, heart_wav, audio_result):
    monkeypatch.setattr(gateway, "AUDIO_ENGINE_URL", "http://audio-engine:8001/predict")
    monkeypatch.setattr(gateway, "ORCHESTRATOR_URL", "")
    monkeypatch.setattr(gateway.requests, "post", lambda url, **kwargs: FakeResponse(200, audio_result))
    data = client.post("/analyze_pcg", files={"file": ("heart.wav", heart_wav, "audio/wav")}).json()
    assert data == audio_result


def test_forwards_the_wav_then_the_audio_result(monkeypatch, heart_wav, audio_result):
    monkeypatch.setattr(gateway, "AUDIO_ENGINE_URL", "http://audio-engine:8001/predict")
    monkeypatch.setattr(gateway, "ORCHESTRATOR_URL", "http://agent-orchestrator:8002/orchestrate")
    calls = []

    def fake_post(url, files=None, json=None, timeout=None):
        calls.append({"url": url, "files": files, "json": json, "timeout": timeout})
        if url.endswith("/predict"):
            return FakeResponse(200, audio_result)
        return FakeResponse(200, {"audio": json, "report": {"summary": "ok"}})

    monkeypatch.setattr(gateway.requests, "post", fake_post)
    data = client.post("/analyze_pcg", files={"file": ("heart.wav", heart_wav, "audio/wav")}).json()

    assert calls[0]["url"].endswith("/predict") and calls[0]["files"]["file"][1] == heart_wav    # 1. the wav
    assert calls[1]["url"].endswith("/orchestrate") and calls[1]["json"] == audio_result         # 2. the result
    assert calls[1]["timeout"] >= 300                                     # the LLM can take minutes
    assert data["report"] == {"summary": "ok"}


def test_orchestrator_down_still_gives_the_audio_result(monkeypatch, heart_wav, audio_result):
    monkeypatch.setattr(gateway, "AUDIO_ENGINE_URL", "http://audio-engine:8001/predict")
    monkeypatch.setattr(gateway, "ORCHESTRATOR_URL", "http://agent-orchestrator:8002/orchestrate")

    def fake_post(url, **kwargs):
        if url.endswith("/predict"):
            return FakeResponse(200, audio_result)
        raise requests.Timeout("too slow")

    monkeypatch.setattr(gateway.requests, "post", fake_post)
    response = client.post("/analyze_pcg", files={"file": ("heart.wav", heart_wav, "audio/wav")})
    assert response.status_code == 200
    data = response.json()
    assert data["audio"] == audio_result and data["report"] is None
    assert "indisponible" in data["report_error"]


def test_get_duration_of_a_wav_and_of_a_wrong_file(heart_wav):
    assert gateway.get_duration(heart_wav) == 6.0
    assert gateway.get_duration(b"not a wav") is None
