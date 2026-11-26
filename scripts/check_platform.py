"""
End-to-end check of the running platform with REAL wavs of the TEST set (idea from Zaineb's check_api.py).

    venv\\Scripts\\python scripts\\check_platform.py            20 abnormal + 20 normal test wavs -> the audio engine
    venv\\Scripts\\python scripts\\check_platform.py 5          5 + 5 wavs
    venv\\Scripts\\python scripts\\check_platform.py 1 --agents 1 + 1 wavs through the gateway (audio + agents, slow)

The test wavs (splits/test_files.json) were never used to train the models, so the recall and the specificity
printed here are an honest end-to-end measure of the deployed platform (Docker, ONNX Runtime, the same features).
"""
import json
import os
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "ourdataset", "train")
ENGINE_URL = os.getenv("ENGINE_URL", "http://localhost:8001/predict")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000/analyze_pcg")

n = 20
if len(sys.argv) > 1:
    n = int(sys.argv[1])
use_agents = "--agents" in sys.argv
url = GATEWAY_URL if use_agents else ENGINE_URL

with open(os.path.join(ROOT, "splits", "test_files.json")) as f:
    split = json.load(f)

truth = []
predictions = []
seconds = []
for label, folder in [(1, "abnormal"), (0, "normal")]:
    names = [name for name, l in zip(split["test_files"], split["test_labels"]) if l == label][:n]
    for name in names:
        with open(os.path.join(DATASET, folder, name), "rb") as wav:
            start = time.time()
            response = requests.post(url, files={"file": (name, wav, "audio/wav")}, timeout=1200)
            seconds.append(time.time() - start)
        if response.status_code != 200:
            print("error", response.status_code, name, response.text[:200])
            continue
        data = response.json()
        audio = data.get("audio", data)          # the gateway with agents answers {"audio": ..., "report": ...}
        truth.append(label)
        predictions.append(1 if audio["prediction"] == "abnormal" else 0)
        if use_agents and data.get("report"):
            report = data["report"]
            print(name, "->", audio["prediction"], "| risk", report["risk_level"], "| approved",
                  report["audit"]["approved"], "|", len(report["sources"]), "sources |", round(seconds[-1], 1), "s")

tp = sum(1 for t, p in zip(truth, predictions) if t == 1 and p == 1)
fn = sum(1 for t, p in zip(truth, predictions) if t == 1 and p == 0)
tn = sum(1 for t, p in zip(truth, predictions) if t == 0 and p == 0)
fp = sum(1 for t, p in zip(truth, predictions) if t == 0 and p == 1)
print("wavs:", len(truth), "| url:", url)
print("abnormal found as abnormal: {} of {} (recall {:.2f})".format(tp, tp + fn, tp / max(tp + fn, 1)))
print("normal found as normal:     {} of {} (specificity {:.2f})".format(tn, tn + fp, tn / max(tn + fp, 1)))
print("mean time per wav: {:.2f} s".format(sum(seconds) / max(len(seconds), 1)))
