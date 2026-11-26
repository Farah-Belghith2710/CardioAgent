"""
Service 4 : agent-orchestrator (FastAPI + the LangGraph graph of the 3 agents)

  api-gateway --(the JSON of the audio engine)--> POST /orchestrate --> {"audio": ..., "report": CardioReport}

Guardrails "sans fuite de données": everything stays in our Docker network (Ollama for the LLM, Qdrant for the RAG,
no external API), the audio never reaches the LLM (only the numbers of the audio engine), the file name is removed,
and nothing about the patient is stored. Each run logs only metrics to MLflow (experiment "llm_inference").
"""
import os

import requests
from fastapi import BackgroundTasks, FastAPI

from agents import llm
from agents.graph import run_agents
from agents.schemas import AudioResult
from rag_engine.ingest import QDRANT_URL

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")    # if MLflow is down: give up fast
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")


def log_llm_inference(metrics):
    # the inference metrics of the LLM in MLflow: latency of each agent, tokens, schema valid, approved, retries.
    # Never the text of the report (no patient data). A problem with MLflow never breaks a report.
    if MLFLOW_TRACKING_URI == "":
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("llm_inference")
        with mlflow.start_run(run_name="orchestrate"):
            mlflow.log_param("llm_model", llm.LLM_MODEL)
            mlflow.log_param("risk_level", metrics.get("risk_level", ""))
            for name in ["analyste_s", "cardiologue_s", "auditeur_s", "total_s", "prompt_tokens", "eval_tokens",
                         "retries", "n_sources"]:
                mlflow.log_metric(name, metrics.get(name, 0))
            for name in ["schema_valid", "approved", "fallback"]:
                mlflow.log_metric(name, int(bool(metrics.get(name, False))))    # True / False -> 1 / 0
    except Exception as error:
        print("MLflow logging failed (the report is not affected):", error)


app = FastAPI(title="agent-orchestrator",
              description="Service 4 : orchestrateur multi-agents (Analyste, Cardiologue RAG, Auditeur)")


@app.get("/health")
def health():
    try:
        qdrant_ok = requests.get(QDRANT_URL + "/readyz", timeout=3).status_code == 200
    except requests.RequestException:
        qdrant_ok = False
    return {"status": "ok", "llm_model": llm.LLM_MODEL, "llm_ready": llm.is_ready(), "qdrant_ready": qdrant_ok}


@app.post("/orchestrate")
def orchestrate(audio: AudioResult, background_tasks: BackgroundTasks):
    # FastAPI checks the JSON with AudioResult (Pydantic) before this function: a wrong JSON -> error 422
    report, metrics = run_agents(audio.model_dump())
    background_tasks.add_task(log_llm_inference, metrics)    # after the answer is sent
    return {"audio": audio.model_dump(), "report": report, "agents_metrics": metrics}
