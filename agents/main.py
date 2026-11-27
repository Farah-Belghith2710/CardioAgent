"""
Service 4 : agent-orchestrator (FastAPI + the LangGraph graph of the 3 agents)

  api-gateway --(the JSON of the audio engine)--> POST /orchestrate --> {"audio": ..., "report": CardioReport}
  api-gateway --(question + last analysis)-------> POST /chat        --> ChatAnswer (agents/chat.py)

Guardrails "sans fuite de données": in the local mode everything stays in our Docker network (Ollama for the LLM,
Qdrant for the RAG). In the online mode (Groq / xAI, agents/llm.py) only the question and the numbers of the analysis
go to the provider. In both modes the audio never reaches the LLM, the file name is removed, and nothing about the
patient is stored. Each run logs only metrics to MLflow (experiment "llm_inference"): never the texts, never the key.
"""
import os

import requests
from fastapi import BackgroundTasks, FastAPI

from agents import llm
from agents.chat import chat
from agents.graph import run_agents
from agents.schemas import AudioResult, ChatAnswer, ChatRequest
from rag_engine.ingest import QDRANT_URL

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")      # if MLflow is down: give up after 2 retries
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")   # (the logging runs after the answer is sent)
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")


def log_to_mlflow(run_name, params, metrics):
    # 1 small run in the MLflow experiment "llm_inference" (latency, tokens, ...). Never the text of the report or
    # of the chat (no patient data). A problem with MLflow never breaks an answer.
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
        run_id = client.create_run(experiment_id, run_name=run_name).info.run_id
        client.log_param(run_id, "llm_provider", llm.provider())    # groq / xai / ollama (never the key)
        client.log_param(run_id, "llm_model", llm.model_name())
        for name in params:
            client.log_param(run_id, name, params[name])
        for name in metrics:
            client.log_metric(run_id, name, metrics[name])
        client.set_terminated(run_id)
    except Exception as error:
        print("MLflow logging failed (the answer is not affected):", error)


def log_llm_inference(metrics):
    # the report of the 3 agents: latency of each agent, tokens, schema valid, approved, retries
    values = {}
    for name in ["analyste_s", "cardiologue_s", "auditeur_s", "total_s", "prompt_tokens", "eval_tokens",
                 "retries", "n_sources"]:
        values[name] = metrics.get(name, 0)
    for name in ["schema_valid", "approved", "fallback"]:
        values[name] = int(bool(metrics.get(name, False)))    # True / False -> 1 / 0
    log_to_mlflow("orchestrate", {"risk_level": metrics.get("risk_level", "")}, values)


def log_chat(metrics):
    # 1 question of the chat: latency, on topic or not (and how it was decided), tokens, number of sources
    values = {"total_s": metrics.get("total_s", 0), "on_topic": int(bool(metrics.get("on_topic"))),
              "prompt_tokens": metrics.get("prompt_tokens", 0), "eval_tokens": metrics.get("eval_tokens", 0),
              "n_sources": metrics.get("n_sources", 0), "safe_answer": int(bool(metrics.get("safe_answer")))}
    log_to_mlflow("chat", {"guardrail": metrics.get("guardrail", "")}, values)


app = FastAPI(title="agent-orchestrator",
              description="Service 4 : orchestrateur multi-agents (Analyste, Cardiologue RAG, Auditeur)")


@app.get("/health")
def health():
    try:
        qdrant_ok = requests.get(QDRANT_URL + "/readyz", timeout=3).status_code == 200
    except requests.RequestException:
        qdrant_ok = False
    # llm_provider: the page shows "Mode en ligne" (groq / xai) or "Mode local" (ollama) in its footer
    return {"status": "ok", "llm_provider": llm.provider(), "llm_online": llm.is_online(), "llm_model": llm.model_name(),
            "llm_ready": llm.is_ready(), "qdrant_ready": qdrant_ok}


@app.post("/orchestrate")
def orchestrate(audio: AudioResult, background_tasks: BackgroundTasks):
    # FastAPI checks the JSON with AudioResult (Pydantic) before this function: a wrong JSON -> error 422
    report, metrics = run_agents(audio.model_dump())
    background_tasks.add_task(log_llm_inference, metrics)    # after the answer is sent
    return {"audio": audio.model_dump(), "report": report, "agents_metrics": metrics}


@app.post("/chat", response_model=ChatAnswer)
def chat_route(request: ChatRequest, background_tasks: BackgroundTasks):
    # a question about the analysed wav or about the heart (FastAPI checks the JSON with ChatRequest -> 422 if wrong)
    history = [message.model_dump() for message in request.history]
    answer, metrics = chat(request.question, request.analysis, history)
    background_tasks.add_task(log_chat, metrics)
    return answer
