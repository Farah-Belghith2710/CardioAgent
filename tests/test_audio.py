"""
Unit tests of the audio engine (Service 3): signal transformations, features of each model, quality rules, /predict.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import audio_engine.main as engine

SR = engine.SR
client = TestClient(engine.app)


# ==================== cut(): the same duration for every wav ====================
def test_cut_keeps_the_middle_of_a_long_signal():
    signal = np.arange(SR * 20, dtype=np.float32)          # 20 s
    part = engine.cut(signal, 5)
    assert len(part) == SR * 5
    start = (SR * 20 - SR * 5) // 2                         # the middle 5 s
    assert part[0] == signal[start] and part[-1] == signal[start + SR * 5 - 1]


def test_cut_repeats_a_short_signal():
    signal = np.arange(SR * 2, dtype=np.float32)            # 2 s -> repeated until 5 s
    part = engine.cut(signal, 5)
    assert len(part) == SR * 5
    assert np.array_equal(part[:SR * 2], signal) and np.array_equal(part[SR * 2:SR * 4], signal)


# ==================== the features of each model: the shapes of the notebooks ====================
@pytest.fixture
def signal():
    t = np.arange(SR * 8) / SR
    return (0.5 * np.sin(2 * np.pi * 60 * t) * (np.sin(2 * np.pi * 1.2 * t) > 0.6)).astype(np.float32) + 0.01


def test_features_tree_is_26_mfcc_numbers(signal):
    X = engine.features_tree(signal)
    assert X.shape == (1, 26) and X.dtype == np.float32 and np.isfinite(X).all()


def test_features_extended_is_72_numbers(signal):
    X = engine.features_extended(signal)
    assert X.shape == (1, 72) and np.isfinite(X).all()
    assert X[0, 0] == pytest.approx(8.0)                    # the 1st feature = the duration in seconds


def test_features_svm_is_the_flattened_mel_spectrogram(signal):
    X = engine.features_svm(signal, 5)
    assert X.shape == (1, 30 * 40)                          # 30 Mel bands x 40 moments (5 s, hop 256)


def test_features_cnn_and_transformer_are_40x157_images_from_0_to_1(signal):
    for X in [engine.features_cnn(signal), engine.features_transformer(signal)]:
        assert X.shape == (1, 40, 157, 1)
        assert X.min() >= 0 and X.max() <= 1


def test_features_resnet_is_a_magma_image(signal):
    if engine.MAGMA_LUT is None:
        pytest.skip("magma_lut.npy not found (made by notebook 05)")
    X = engine.features_resnet(signal)
    assert X.shape == (1, 40, 157, 3)
    assert X.min() >= 0 and X.max() <= 255


def test_magma_table_has_256_colours():
    if engine.MAGMA_LUT is None:
        pytest.skip("magma_lut.npy not found (made by notebook 05)")
    assert engine.MAGMA_LUT.shape == (256, 3)
    assert engine.MAGMA_LUT.min() >= 0 and engine.MAGMA_LUT.max() <= 1


def test_model_input_chooses_the_features_of_each_model(signal):
    assert engine.model_input({"model": "SVM", "features": "extended features (72)", "seconds": "whole wav"},
                              signal).shape == (1, 72)
    assert engine.model_input({"model": "DecisionTree", "features": "MFCC mean + std (26)", "seconds": 5},
                              signal).shape == (1, 26)
    assert engine.model_input({"model": "SVM", "features": "mel spectrogram flatten (30 bands)", "seconds": 5},
                              signal).shape == (1, 1200)
    assert engine.model_input({"model": "CNN", "features": "", "seconds": 5}, signal).shape == (1, 40, 157, 1)


# ==================== the API ====================
def test_health_lists_the_loaded_models():
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert len(data["models"]) >= 1
    assert set(data["models"]) | set(data["missing_models"]) == {m["model"] for m in engine.benchmark["models"]}


def test_benchmark_returns_benchmark_json():
    data = client.get("/benchmark").json()
    assert data["best_model"] == engine.benchmark["best_model"]
    assert len(data["models"]) == len(engine.benchmark["models"])


def test_predict_gives_the_answer_of_every_model(heart_wav):
    response = client.post("/predict", files={"file": ("heart.wav", heart_wav, "audio/wav")})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["normal", "abnormal"]
    assert data["model"] == engine.models[0]["model"]                  # the official answer = the best model
    assert len(data["benchmark"]) == len(engine.models)
    assert data["duration_s"] == pytest.approx(6.0)
    assert data["warning"] is None
    for answer in data["benchmark"]:
        assert answer["prediction"] in ["normal", "abnormal"]
        if answer["probability_abnormal"] is not None:
            assert 0 <= answer["probability_abnormal"] <= 1


def test_predict_warns_for_a_short_wav(short_wav):
    data = client.post("/predict", files={"file": ("short.wav", short_wav, "audio/wav")}).json()
    assert "court" in data["warning"]


def test_predict_warns_for_a_saturated_wav(saturated_wav):
    data = client.post("/predict", files={"file": ("loud.wav", saturated_wav, "audio/wav")}).json()
    assert "saturé" in data["warning"]


def test_predict_refuses_a_silent_wav(silent_wav):
    response = client.post("/predict", files={"file": ("silent.wav", silent_wav, "audio/wav")})
    assert response.status_code == 400
    assert "silencieux" in response.json()["detail"]


def test_predict_refuses_a_file_that_is_not_audio():
    response = client.post("/predict", files={"file": ("notes.wav", b"this is not a wav file", "audio/wav")})
    assert response.status_code == 400


def test_run_ensemble_is_the_mean_of_its_members():
    m = {"model": "Ensemble", "members": ["CNN", "ResNet50"], "recall": 0.9, "precision": 0.6, "f1": 0.7,
         "accuracy": 0.8}
    answers = {"CNN": {"probability_abnormal": 0.8}, "ResNet50": {"probability_abnormal": 0.3}}
    answer = engine.run_ensemble(m, answers)
    assert answer["probability_abnormal"] == 0.55 and answer["prediction"] == "abnormal"
    answers["CNN"]["probability_abnormal"] = 0.6
    assert engine.run_ensemble(m, answers)["prediction"] == "normal"         # (0.6 + 0.3) / 2 = 0.45


# ==================== train_all.py: runs the training notebooks in the right order ====================
def test_train_all_runs_the_notebooks_in_order(monkeypatch):
    import os
    import runpy
    import subprocess
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, check=False: commands.append(command))
    runpy.run_path(os.path.join(os.path.dirname(engine.HERE), "audio_engine", "train_all.py"), run_name="__main__")
    notebooks = [os.path.basename(command[-1]) for command in commands]
    assert notebooks == ["02_Tp3_svm.ipynb", "03_Tp4_decision_tree.ipynb", "04_Tp7_cnn.ipynb",
                         "06_Tp8_transfer_learning.ipynb", "07_transformer_cnn.ipynb", "08_ensemble.ipynb",
                         "05_best_model_onnx.ipynb"]
    assert "--execute" in commands[0] and "--inplace" in commands[0]


# ==================== inference logs in MLflow: never break a prediction ====================
def test_log_inference_does_nothing_without_mlflow(monkeypatch):
    monkeypatch.setattr(engine, "MLFLOW_TRACKING_URI", "")
    engine.log_inference({"model": "CNN"}, 12.0)    # no error, no call


def test_log_inference_survives_an_mlflow_error(monkeypatch):
    monkeypatch.setattr(engine, "MLFLOW_TRACKING_URI", "http://127.0.0.1:9")    # nothing listens on port 9
    result = {"model": "CNN", "prediction": "abnormal", "probability_abnormal": 0.9, "warning": None,
              "duration_s": 6.0, "benchmark": []}
    engine.log_inference(result, 12.0)               # MLflow unreachable (or not installed): no exception


def test_health_answers_while_the_warm_up_is_running(monkeypatch):
    # the warm-up runs in a background thread: /health must answer at once, with "warming_up": true
    import time

    def slow_model(m, signal):
        time.sleep(0.3)    # a slow model (like ResNet50 on a busy PC)

    monkeypatch.setattr(engine, "run_model", slow_model)
    engine.warmed_up.clear()
    thread = engine.start_warm_up()
    start = time.time()
    data = client.get("/health").json()
    assert time.time() - start < 2 and data["status"] == "ok" and data["warming_up"] is True
    thread.join(timeout=30)
    assert client.get("/health").json()["warming_up"] is False
