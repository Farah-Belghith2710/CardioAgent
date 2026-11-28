"""
Shared helpers of the tests (pytest reads this file first).
The tests need no dataset: the wavs are made here with numpy (a sine + a little noise) and soundfile.
All the other services (audio engine, orchestrator, Ollama, Qdrant, MLflow) are replaced by fakes (monkeypatch).
"""
import io
import os
import sys

import numpy as np
import pytest
import soundfile as sf

# no automatic warm-up of the audio engine in the tests (a test starts its own, with a slow fake model)
os.environ.setdefault("AUDIO_WARM_UP", "0")

# the repo root, so that "import audio_engine.main", "import api.main", "import agents.graph" work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SR = 2000

# FastAPI's TestClient is a class whose name starts with "Test": pytest must not try to collect it as a test
from fastapi.testclient import TestClient  # noqa: E402
TestClient.__test__ = False

from agents import llm  # noqa: E402


@pytest.fixture(autouse=True)
def no_llm_settings(monkeypatch):
    # every test starts in the local mode (no key), whatever the settings of the computer: the tests never call an
    # online LLM and never need a key (the tests of the online mode set their own fake key)
    for name in ["LLM_PROVIDER", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"]:
        monkeypatch.setattr(llm, name, "")
    monkeypatch.setattr(llm, "_online", {"checked_at": 0.0, "ready": False, "model": None})


def make_wav(seconds=6.0, kind="heart", sr=SR):
    # a small wav in memory (bytes): "heart" = a 60 Hz sine with beats + noise, "silent" = zeros,
    # "saturated" = a square wave at the maximum (all the samples at +1 or -1)
    t = np.arange(int(seconds * sr)) / sr
    if kind == "silent":
        signal = np.zeros_like(t)
    elif kind == "saturated":
        signal = np.sign(np.sin(2 * np.pi * 5 * t))
    else:
        beats = (np.sin(2 * np.pi * 1.2 * t) > 0.6).astype(float)            # ~72 beats per minute
        rng = np.random.default_rng(0)
        signal = 0.5 * beats * np.sin(2 * np.pi * 60 * t) + 0.02 * rng.standard_normal(len(t))
    buffer = io.BytesIO()
    sf.write(buffer, signal.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture
def heart_wav():
    return make_wav(6.0, "heart")


@pytest.fixture
def short_wav():
    return make_wav(1.5, "heart")


@pytest.fixture
def silent_wav():
    return make_wav(6.0, "silent")


@pytest.fixture
def saturated_wav():
    return make_wav(6.0, "saturated")


@pytest.fixture
def audio_result():
    # an answer of the audio engine (the same keys as POST /predict)
    return {
        "filename": "patient_042.wav",
        "duration_s": 12.5,
        "prediction": "abnormal",
        "probability_abnormal": 0.86,
        "model": "CNN",
        "warning": None,
        "benchmark": [
            {"model": "CNN", "prediction": "abnormal", "probability_abnormal": 0.86, "recall": 0.94, "f1": 0.63},
            {"model": "SVM", "prediction": "abnormal", "probability_abnormal": None, "recall": 0.89, "f1": 0.7},
            {"model": "TransformerCNN", "prediction": "abnormal", "probability_abnormal": 0.77},
            {"model": "ResNet50", "prediction": "normal", "probability_abnormal": 0.41},
            {"model": "DecisionTree", "prediction": "abnormal", "probability_abnormal": 0.9},
        ],
    }
