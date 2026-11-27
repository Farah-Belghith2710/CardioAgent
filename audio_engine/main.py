"""
Service 3 : audio-engine (ONNX Runtime + Librosa)

The benchmark of our 5 models (exported to ONNX by the notebook 05) on each heart sound (.wav):
  - every model gets the wav prepared EXACTLY like in its notebook (TP4 tree, TP3 SVM, TP7 CNN, TP8 ResNet50,
    07 Transformer-CNN)
  - the ensemble of notebook 08 = the mean of the probabilities of its members (no extra ONNX file)
  - the official answer = the best model of the benchmark (the highest mean validation recall in MLflow)
  - each prediction is logged in MLflow (experiment "inference_logs") when MLFLOW_TRACKING_URI is set
  api-gateway --(.wav)--> POST /predict --> the official answer + the answer and the scores of each model
"""
import io
import json
import os
import time

import librosa
import numpy as np
import onnxruntime as ort
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

SR = 2000    # every wav at 2000 Hz (like in all the notebooks)
HERE = os.path.dirname(os.path.abspath(__file__))    # the folder of this file (the ONNX files are next to it)

# MLflow (Service 6) for the traceability of the predictions. Empty = no logging (tests, CI, MLflow not ready).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")      # if MLflow is down: give up after 2 retries
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")   # (the logging runs after the answer is sent)
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

# benchmark.json (written by the notebook 05): our models, their ONNX file and their scores in MLflow,
# sorted by validation recall -> the first one is the best one
with open(os.path.join(HERE, "benchmark.json")) as f:
    benchmark = json.load(f)

# the 256 colours of plt.cm.magma (saved by notebook 05): to make the image of ResNet50 without matplotlib
MAGMA_LUT = None
if os.path.exists(os.path.join(HERE, "magma_lut.npy")):
    MAGMA_LUT = np.load(os.path.join(HERE, "magma_lut.npy"))

# the ONNX models are loaded ONCE, when the container starts (not at each request).
# A missing file is skipped with a warning (resnet50.onnx is ~95 MB, too big for git: the notebook 05 makes it),
# so the engine also starts in the CI and after a fresh clone.
sessions = {}
models = []            # the models of the benchmark that are really loaded, in the benchmark order
missing_models = []
for m in benchmark["models"]:
    if m["model"] == "Ensemble":
        continue           # the ensemble has no file: it is checked just below
    path = os.path.join(HERE, m["file"])
    if os.path.exists(path) and (m["model"] != "ResNet50" or MAGMA_LUT is not None):
        sessions[m["model"]] = ort.InferenceSession(path)
    else:
        missing_models.append(m["model"])
        print("WARNING:", m["file"], "not found -> the model", m["model"], "is skipped (run notebook 05 to make it)")

# the ensemble (notebook 08) = the mean of the probabilities of its members: it works only if all of them are loaded
for m in benchmark["models"]:
    if m["model"] == "Ensemble":
        ready = True
        for member in m["members"]:
            if member not in sessions:
                ready = False
        if ready:
            models.append(m)
        else:
            missing_models.append("Ensemble")
    elif m["model"] in sessions:
        models.append(m)


def cut(signal, seconds):
    # the same duration for every wav (like in the notebooks): the middle part, a shorter wav is repeated
    n = SR * seconds
    if len(signal) < n:
        return np.tile(signal, n // len(signal) + 1)[:n]
    start = (len(signal) - n) // 2
    return signal[start:start + n]


def signaltonoise(a, axis=0, ddof=0):
    # the SNR function of the course (TP1): mean / standard deviation
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


# the features of each model = the SAME code as in its notebook
def features_tree(signal):
    # TP4: 13 MFCC on the middle 5 s -> the mean and the std of each one = 26 numbers
    mfcc = librosa.feature.mfcc(y=cut(signal, 5), sr=SR, n_mfcc=13, n_fft=256, hop_length=128, n_mels=30)
    features = list(np.mean(mfcc, axis=1)) + list(np.std(mfcc, axis=1))
    return np.array([features], dtype=np.float32)                         # 1 wav x 26 numbers


def features_extended(signal):
    # Zaineb's 72 features on the whole wav (notebooks 02 and 03): duration, SNR, mean and std of 20 Mel bands,
    # mean and std of 13 MFCC, mean and std of the spectral centroid and of the zero-crossing rate
    duration = len(signal) / SR
    snr = float(signaltonoise(signal / max(np.max(np.abs(signal)), 1e-10)))
    S = librosa.feature.melspectrogram(y=signal, sr=SR, n_fft=512, hop_length=128, n_mels=20)
    S_DB = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=13, n_fft=512, hop_length=128, n_mels=20)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=SR, n_fft=512, hop_length=128)
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=512, hop_length=128)
    features = ([duration, snr] + list(np.mean(S_DB, axis=1)) + list(np.std(S_DB, axis=1))
                + list(np.mean(mfcc, axis=1)) + list(np.std(mfcc, axis=1))
                + [np.mean(centroid), np.std(centroid), np.mean(zcr), np.std(zcr)])
    return np.array([features], dtype=np.float32)                         # 1 wav x 72 numbers


def features_svm(signal, seconds):
    # TP3: Mel spectrogram of 30 bands, flattened
    S = librosa.feature.melspectrogram(y=cut(signal, seconds), sr=SR, n_fft=256, hop_length=256, n_mels=30)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return np.array([S_DB.flatten()], dtype=np.float32)                   # 1 wav x (30 bands x N moments)


def spectrogram_40x157(signal):
    # TP7: Mel spectrogram of 40 bands on the middle 5 s = an image of 40 x 157, with values from 0 to 1
    S = librosa.feature.melspectrogram(y=cut(signal, 5), sr=SR, n_fft=256, hop_length=64, n_mels=40)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return (S_DB + 80) / 80


def features_cnn(signal):
    return spectrogram_40x157(signal).reshape(1, 40, 157, 1).astype(np.float32)    # 1 wav x 40 x 157 x 1 channel


def features_transformer(signal):
    # 07: the same input as the CNN (the CNN front-end and the Transformer blocks are inside the ONNX model)
    return spectrogram_40x157(signal).reshape(1, 40, 157, 1).astype(np.float32)


def features_resnet(signal):
    # TP8: the spectrogram coloured with magma (like plt.cm.magma: colour number = int(x * 256)), colours 0 to 255.
    # The resize to 224 x 224 and preprocess_input are INSIDE the ONNX model (notebook 05).
    S = spectrogram_40x157(signal)
    index = np.minimum((S * 256).astype(int), 255)
    image = MAGMA_LUT[index] * 255
    return image.reshape(1, 40, 157, 3).astype(np.float32)               # 1 wav x 40 x 157 x 3 colours


def model_input(m, signal):
    # which features for which model (the "features" text comes from MLflow, through benchmark.json)
    if m["model"] in ["DecisionTree", "SVM"] and "extended" in m["features"]:
        return features_extended(signal)
    if m["model"] == "DecisionTree":
        return features_tree(signal)
    if m["model"] == "SVM":
        return features_svm(signal, int(m["seconds"]))
    if m["model"] == "CNN":
        return features_cnn(signal)
    if m["model"] == "TransformerCNN":
        return features_transformer(signal)
    return features_resnet(signal)


def run_model(m, signal):
    # 1 model of the benchmark on the wav -> its answer, with its scores from MLflow
    session = sessions[m["model"]]
    X = model_input(m, signal)
    outputs = session.run(None, {session.get_inputs()[0].name: X})

    if m["model"] in ["CNN", "ResNet50", "TransformerCNN"]:
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
        "recall": m["recall"],                   # validation (mean of the 3 folds)
        "precision": m["precision"],
        "f1": m["f1"],
        "accuracy": m["accuracy"],
        "test_recall": m.get("test_recall"),     # test (the 20% test set)
        "test_f1": m.get("test_f1"),
    }


def run_ensemble(m, answers_by_model):
    # notebook 08 (soft vote): the mean of the probabilities of "abnormal" of the members, more than 0.5 -> abnormal
    total = 0
    for member in m["members"]:
        total = total + answers_by_model[member]["probability_abnormal"]
    probability = total / len(m["members"])
    if probability > 0.5:
        prediction = "abnormal"
    else:
        prediction = "normal"
    return {
        "model": "Ensemble",
        "prediction": prediction,
        "probability_abnormal": round(probability, 3),
        "recall": m["recall"],
        "precision": m["precision"],
        "f1": m["f1"],
        "accuracy": m["accuracy"],
        "test_recall": m.get("test_recall"),
        "test_f1": m.get("test_f1"),
    }


def log_inference(result, latency_ms):
    # traceability (PDF, Service 6): 1 small MLflow run per prediction, in the experiment "inference_logs".
    # No file name and no audio: no patient data leaves the engine. A problem with MLflow never breaks a prediction.
    if MLFLOW_TRACKING_URI == "":
        return
    try:
        # MlflowClient = the explicit MLflow client: no shared "active run", safe in a background task
        from mlflow.tracking import MlflowClient
        client = MlflowClient(MLFLOW_TRACKING_URI)
        experiment = client.get_experiment_by_name("inference_logs")
        if experiment is None:
            experiment_id = client.create_experiment("inference_logs")
        else:
            experiment_id = experiment.experiment_id
        run_id = client.create_run(experiment_id, run_name="predict_" + result["model"]).info.run_id
        client.log_param(run_id, "model", result["model"])
        client.log_param(run_id, "prediction", result["prediction"])
        client.log_param(run_id, "warning", result["warning"] or "none")
        if result["probability_abnormal"] is not None:
            client.log_metric(run_id, "probability_abnormal", result["probability_abnormal"])
        client.log_metric(run_id, "latency_ms", latency_ms)
        client.log_metric(run_id, "duration_s", result["duration_s"])
        client.log_metric(run_id, "n_models", len(result["benchmark"]))
        client.set_terminated(run_id)
    except Exception as error:
        print("MLflow logging failed (the prediction is not affected):", error)


# warm-up: the first call of librosa compiles some of its functions (numba) and ONNX Runtime prepares each model.
# We do it once now, when the container starts, so the first real request is not slow.
warm_up_signal = np.random.default_rng(0).normal(0, 0.1, SR * 6).astype(np.float32)
for m in models:
    if m["model"] != "Ensemble":
        run_model(m, warm_up_signal)


app = FastAPI(title="audio-engine", description="Service 3 : moteur audio (benchmark de nos 5 modèles au format ONNX)")


@app.get("/health")
def health():
    # quick check that the container is running, and which models it has
    return {"status": "ok", "models": [m["model"] for m in models], "missing_models": missing_models,
            "best_model": benchmark["best_model"], "mlflow_logging": MLFLOW_TRACKING_URI != ""}


@app.get("/benchmark")
def get_benchmark():
    # the benchmark of our models: their validation and test scores (from MLflow), the best one first
    return benchmark


@app.post("/predict")
def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    start = time.time()
    if len(models) == 0:
        raise HTTPException(status_code=503, detail="Aucun modèle ONNX chargé : lancez le notebook 05")

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

    # 3. the benchmark: every model gives its answer, then the ensemble (the mean of the probabilities of its members)
    answers_by_model = {}
    for m in models:
        if m["model"] != "Ensemble":
            answers_by_model[m["model"]] = run_model(m, signal)
    for m in models:
        if m["model"] == "Ensemble":
            answers_by_model["Ensemble"] = run_ensemble(m, answers_by_model)
    answers = []
    for m in models:                  # in the order of the benchmark: the first one = the best model
        answers.append(answers_by_model[m["model"]])
    best = answers[0]

    result = {
        "filename": file.filename,
        "duration_s": round(len(signal) / SR, 2),
        "prediction": best["prediction"],                    # the official answer = the best model
        "probability_abnormal": best["probability_abnormal"],
        "model": best["model"],
        "benchmark": answers,                                # the answer of each model, with its scores
        "warning": warning,
    }
    latency_ms = round((time.time() - start) * 1000, 1)
    background_tasks.add_task(log_inference, result, latency_ms)    # after the answer is sent: no extra waiting
    return result
