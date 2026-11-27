"""
Service 5 : api-gateway (FastAPI + Uvicorn)

Point d'entrée unique de la plateforme :
  client --(.wav)--> POST /analyze_pcg --> moteur audio (Service 3) --> orchestrateur des agents (Service 4)
  client <--(résultat audio + rapport des agents)------------------------'
  POST /analyze_pcg?agents=false -> only the audio result (1 s); then POST /report with it -> the report (minutes)
  POST /chat  -> a question about the analysis or the heart -> the assistant (Service 4)
"""
import io
import os
import wave

import requests
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

# Addresses of the 2 other services. We give them when we start the container (docker-compose.yml):
#   AUDIO_ENGINE_URL=http://audio-engine:8001/predict  ORCHESTRATOR_URL=http://agent-orchestrator:8002/orchestrate
# Empty = that service is not ready yet.
AUDIO_ENGINE_URL = os.getenv("AUDIO_ENGINE_URL", "")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "")
AUDIO_ENGINE_TIMEOUT = 180                                             # seconds (6 models on a busy CPU)
ORCHESTRATOR_TIMEOUT = float(os.getenv("ORCHESTRATOR_TIMEOUT", "900"))  # the LLM on a CPU can take minutes
CHAT_TIMEOUT = float(os.getenv("CHAT_TIMEOUT", "300"))                   # 1 short answer of the LLM

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


def llm_provider():
    # the LLM mode for the footer of the page (groq / xai = online, ollama = local), asked to the orchestrator
    if ORCHESTRATOR_URL == "":
        return None
    try:
        return requests.get(ORCHESTRATOR_URL.rsplit("/", 1)[0] + "/health", timeout=3).json().get("llm_provider")
    except Exception:
        return None    # the orchestrator does not answer yet: the page shows no mode


@app.get("/health")
def health():
    # quick check that the container is running, which services it knows, and the mode of the LLM
    return {"status": "ok", "audio_engine": AUDIO_ENGINE_URL != "", "orchestrator": ORCHESTRATOR_URL != "",
            "llm_provider": llm_provider()}


def chat_url():
    # the chat is on the same service as the agents: http://agent-orchestrator:8002/orchestrate -> .../chat
    return ORCHESTRATOR_URL.rsplit("/", 1)[0] + "/chat"


@app.post("/analyze_pcg")
def analyze_pcg(file: UploadFile = File(...), agents: bool = True):
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

    # 4. send the audio engine's result to the orchestrator of the agents (if it's ready and asked)
    #    agents=false: the web page shows the audio result at once, then asks for the report with /report
    if ORCHESTRATOR_URL == "" or not agents:
        return audio_result
    try:
        return send_to_service(ORCHESTRATOR_URL, json=audio_result, timeout=ORCHESTRATOR_TIMEOUT)
    except HTTPException as error:
        # the agents are down: the client still gets the result of the audio engine (degraded mode)
        return {"audio": audio_result, "report": None,
                "report_error": "Rapport des agents indisponible : " + str(error.detail)}


@app.post("/report")
def report(audio_result: dict = Body(...)):
    # the report of the 3 agents for an audio result already shown on the page (the web page: step 2)
    if ORCHESTRATOR_URL == "":
        raise HTTPException(status_code=503, detail="L'orchestrateur des agents n'est pas connecté (ORCHESTRATOR_URL vide)")
    return send_to_service(ORCHESTRATOR_URL, json=audio_result, timeout=ORCHESTRATOR_TIMEOUT)


@app.post("/chat")
def chat(question: dict = Body(...)):
    # the chat: {"question", "analysis", "history"} -> the assistant of Service 4 (it checks the JSON: 422 if wrong)
    if ORCHESTRATOR_URL == "":
        raise HTTPException(status_code=503, detail="L'assistant n'est pas connecté (ORCHESTRATOR_URL vide)")
    return send_to_service(chat_url(), json=question, timeout=CHAT_TIMEOUT)
