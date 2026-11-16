"""
Service 3 : audio-engine (ONNX Runtime + Librosa)

The benchmark of our models (exported to ONNX by the notebook 05) on each heart sound (.wav):
  - every model gets the wav prepared EXACTLY like in its notebook (TP4 tree, TP3 SVM, TP7 CNN)
  - the official answer = the best model of the benchmark (the highest recall in MLflow)
  api-gateway --(.wav)--> POST /predict --> the official answer + the answer and the scores of each model
"""
import io
import json

import librosa
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile

SR = 2000    # every wav at 2000 Hz (like in all the notebooks)

# benchmark.json (written by the notebook 05): our models, their ONNX file and their scores in MLflow,
# sorted by recall -> the first one is the best one
with open("benchmark.json") as f:
    benchmark = json.load(f)

# the ONNX models are loaded ONCE, when the container starts (not at each request)
sessions = {}
for m in benchmark["models"]:
    sessions[m["model"]] = ort.InferenceSession(m["file"])


def cut(signal, seconds):
    # the same duration for every wav (like in the notebooks): the middle part, a shorter wav is repeated
    n = SR * seconds
    if len(signal) < n:
        return np.tile(signal, n // len(signal) + 1)[:n]
    start = (len(signal) - n) // 2
    return signal[start:start + n]


# the features of each model = the SAME code as in its notebook
def features_tree(signal, seconds):
    # TP4: 13 MFCC -> the mean and the std of each one = 26 numbers
    mfcc = librosa.feature.mfcc(y=cut(signal, seconds), sr=SR, n_mfcc=13, n_fft=256, hop_length=128, n_mels=30)
    features = list(np.mean(mfcc, axis=1)) + list(np.std(mfcc, axis=1))
    return np.array([features], dtype=np.float32)                         # 1 wav x 26 numbers


def features_svm(signal, seconds):
    # TP3: Mel spectrogram of 30 bands, flattened
    S = librosa.feature.melspectrogram(y=cut(signal, seconds), sr=SR, n_fft=256, hop_length=256, n_mels=30)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return np.array([S_DB.flatten()], dtype=np.float32)                   # 1 wav x (30 bands x N moments)


def features_cnn(signal, seconds):
    # TP7: Mel spectrogram of 40 bands = a grey image of 40 x 157, with values from 0 to 1
    S = librosa.feature.melspectrogram(y=cut(signal, seconds), sr=SR, n_fft=256, hop_length=64, n_mels=40)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return ((S_DB + 80) / 80).reshape(1, 40, 157, 1).astype(np.float32)    # 1 wav x 40 x 157 x 1 channel


FEATURES = {"DecisionTree": features_tree, "SVM": features_svm, "CNN": features_cnn}


def run_model(m, signal):
    # 1 model of the benchmark on the wav -> its answer, with its scores from MLflow
    session = sessions[m["model"]]
    X = FEATURES[m["model"]](signal, m["seconds"])
    outputs = session.run(None, {session.get_inputs()[0].name: X})

    if m["model"] == "CNN":
        probability = float(outputs[0][0][0])    # the sigmoid: the probability of "abnormal"
        label = int(probability > 0.5)
    elif m["model"] == "DecisionTree":
        label = int(outputs[0][0])               # the class (0 or 1)
        probability = float(outputs[1][0][1])    # the % of abnormal wavs in its leaf
    else:
        label = int(outputs[0][0])               # SVM: the class only (trained without probability=True)
        probability = None

    if label == 1:
        prediction = "abnormal"
    else:
        prediction = "normal"
    if probability is not None:
        probability = round(probability, 3)

    return {
        "model": m["model"],
        "prediction": prediction,
        "probability_abnormal": probability,
        "recall": m["recall"],
        "precision": m["precision"],
        "f1": m["f1"],
        "accuracy": m["accuracy"],
    }


app = FastAPI(title="audio-engine", description="Service 3 : moteur audio (benchmark de nos modèles au format ONNX)")


@app.get("/health")
def health():
    # quick check that the container is running, and which models it has
    return {"status": "ok", "models": [m["model"] for m in benchmark["models"]], "best_model": benchmark["best_model"]}


@app.get("/benchmark")
def get_benchmark():
    # the benchmark of our models: their scores on the 3 folds (from MLflow), the best one first
    return benchmark


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    # 1. read the wav, at 2000 Hz
    try:
        signal, rate = librosa.load(io.BytesIO(file.file.read()), sr=SR)
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire ce fichier audio (un .wav est attendu)")

    # 2. the same quality rules as the notebooks
    if len(signal) == 0 or np.max(np.abs(signal)) < 0.001:
        raise HTTPException(status_code=400, detail="Enregistrement silencieux : il n'y a rien à analyser")
    warning = None
    if len(signal) < 2 * SR:
        warning = "Enregistrement très court (moins de 2 s) : résultat moins fiable"
    elif np.mean(np.abs(signal) >= 0.99) > 0.01:
        warning = "Enregistrement saturé : résultat moins fiable"

    # 3. the benchmark: every model gives its answer (the first one = the best model)
    answers = []
    for m in benchmark["models"]:
        answers.append(run_model(m, signal))
    best = answers[0]

    return {
        "filename": file.filename,
        "duration_s": round(len(signal) / SR, 2),
        "prediction": best["prediction"],                    # the official answer = the best model
        "probability_abnormal": best["probability_abnormal"],
        "model": best["model"],
        "benchmark": answers,                                # the answer of each model, with its scores
        "warning": warning,
    }
