"""
Service 5 : api-gateway (FastAPI + Uvicorn)

Point d'entrée unique de la plateforme :
  client --(.wav)--> POST /analyze_pcg --> moteur audio (Service 3) --> orchestrateur des agents (Service 4)
  client <--(résultat audio + rapport des agents)------------------------'
"""
import io
import os
import wave

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

# Addresses of the 2 other services. We give them when we start the container (docker-compose.yml):
#   AUDIO_ENGINE_URL=http://audio-engine:8001/predict  ORCHESTRATOR_URL=http://agent-orchestrator:8002/orchestrate
# Empty = that service is not ready yet.
AUDIO_ENGINE_URL = os.getenv("AUDIO_ENGINE_URL", "")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "")
AUDIO_ENGINE_TIMEOUT = 60                                              # seconds
ORCHESTRATOR_TIMEOUT = float(os.getenv("ORCHESTRATOR_TIMEOUT", "900"))  # the LLM on a CPU can take minutes

HERE = os.path.dirname(os.path.abspath(__file__))


def get_duration(audio):
    # duration of the .wav in seconds = number of samples / sample rate
    try:
        with wave.open(io.BytesIO(audio)) as w:
            return round(w.getnframes() / w.getframerate(), 3)
    except (wave.Error, EOFError):
        return None    # not a readable wav


def send_to_service(url, files=None, json=None, timeout=60):
    # send a POST request to another service and return its JSON answer
    try:
        response = requests.post(url, files=files, json=json, timeout=timeout)
    except requests.RequestException as error:
        # the service is down -> clear message for the client
        raise HTTPException(status_code=503, detail="Service injoignable : " + url + " (" + str(error) + ")")

    if response.status_code != 200:
        # the service answered, but with an error (for example a silent .wav) -> we give its message to the client
        try:
            detail = response.json()["detail"]
        except Exception:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response.json()


app = FastAPI(title="api-gateway", description="Service 5 : point d'entrée unique de la plateforme PCG")


@app.get("/")
def home():
    # the home page = the HTML page in index.html (the /docs test page still exists)
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/health")
def health():
    # quick check that the container is running, and which services it knows
    return {"status": "ok", "audio_engine": AUDIO_ENGINE_URL != "", "orchestrator": ORCHESTRATOR_URL != ""}


@app.post("/analyze_pcg")
def analyze_pcg(file: UploadFile = File(...)):
    # 1. accept only .wav files
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un .wav")
    audio = file.file.read()

    # 2. the audio engine is not ready yet -> just show what we received
    if AUDIO_ENGINE_URL == "":
        return {
            "filename": file.filename,
            "size_bytes": len(audio),
            "duration_s": get_duration(audio),
            "message": "wav reçu. Le moteur audio n'est pas encore connecté (AUDIO_ENGINE_URL vide).",
        }

    # 3. send the .wav to the audio engine
    audio_result = send_to_service(AUDIO_ENGINE_URL, files={"file": (file.filename, audio, "audio/wav")},
                                   timeout=AUDIO_ENGINE_TIMEOUT)

    # 4. send the audio engine's result to the orchestrator of the agents (if it's ready)
    if ORCHESTRATOR_URL == "":
        return audio_result
    try:
        return send_to_service(ORCHESTRATOR_URL, json=audio_result, timeout=ORCHESTRATOR_TIMEOUT)
    except HTTPException as error:
        # the agents are down: the client still gets the result of the audio engine (degraded mode)
        return {"audio": audio_result, "report": None,
                "report_error": "Rapport des agents indisponible : " + str(error.detail)}
