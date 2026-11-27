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
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")      # if MLflow is down: give up after 2 retries
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")   # (the logging runs after the answer is sent)
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")


def log_llm_inference(metrics):
    # the inference metrics of the LLM in MLflow: latency of each agent, tokens, schema valid, approved, retries.
    # Never the text of the report (no patient data). A problem with MLflow never breaks a report.
    if MLFLOW_TRACKING_URI == "":
        return
    try:
        # MlflowClient = the explicit MLflow client: no shared "active run", safe in a background task
        from mlflow.tracking import MlflowClient
        client = MlflowClient(MLFLOW_TRACKING_URI)
        experiment = client.get_experiment_by_name("llm_inference")
        if experiment is None:
            experiment_id = client.create_experiment("llm_inference")
        else:
            experiment_id = experiment.experiment_id
        run_id = client.create_run(experiment_id, run_name="orchestrate").info.run_id
        client.log_param(run_id, "llm_model", llm.LLM_MODEL)
        client.log_param(run_id, "risk_level", metrics.get("risk_level", ""))
        for name in ["analyste_s", "cardiologue_s", "auditeur_s", "total_s", "prompt_tokens", "eval_tokens",
                     "retries", "n_sources"]:
            client.log_metric(run_id, name, metrics.get(name, 0))
        for name in ["schema_valid", "approved", "fallback"]:
            client.log_metric(run_id, name, int(bool(metrics.get(name, False))))    # True / False -> 1 / 0
        client.set_terminated(run_id)
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
