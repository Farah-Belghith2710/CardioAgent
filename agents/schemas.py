"""
The Pydantic (v2) schemas of the agents = the guardrails of the data:
  - AudioResult : the JSON sent by the audio engine (POST /predict), checked before the agents read it
  - CardioReport: the final report. Pydantic refuses a report with a missing field, a wrong type, an unknown
                  prediction or risk level, no recommendation, or a disclaimer that does not send to a doctor.
The fields that come from the audio result (prediction, probability, model, risk_level) are filled by the CODE
(graph.py), never by the LLM: the LLM only writes the text fields (summary, explanation, recommendations, disclaimer).
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ModelAnswer(BaseModel):
    # the answer of 1 model of the benchmark, with its scores in MLflow
    model: str
    prediction: Literal["normal", "abnormal"]
    probability_abnormal: Optional[float] = Field(default=None, ge=0, le=1)
    recall: Optional[float] = None
    precision: Optional[float] = None
    f1: Optional[float] = None
    accuracy: Optional[float] = None
    test_recall: Optional[float] = None
    test_f1: Optional[float] = None


class AudioResult(BaseModel):
    filename: Optional[str] = None
    duration_s: float = Field(ge=0)
    prediction: Literal["normal", "abnormal"]
    probability_abnormal: Optional[float] = Field(default=None, ge=0, le=1)
    model: str
    benchmark: List[ModelAnswer] = Field(min_length=1)
    warning: Optional[str] = None


class Source(BaseModel):
    title: str
    page: int


class Audit(BaseModel):
    approved: bool
    issues: List[str] = []


class CardioReport(BaseModel):
    # filled by the code (from the audio result)
    prediction: Literal["normal", "abnormal"]
    probability_abnormal: Optional[float] = Field(default=None, ge=0, le=1)
    model: str
    risk_level: Literal["faible", "modéré", "élevé"]
    # written by the LLM (Cardiologue agent), checked by the Auditeur agent
    summary: str = Field(min_length=20)
    clinical_explanation: str = Field(min_length=20)
    recommendations: List[str] = Field(min_length=1)
    sources: List[Source]
    disclaimer: str
    audit: Audit

    @field_validator("disclaimer")
    @classmethod
    def disclaimer_sends_to_a_doctor(cls, value):
        lower = value.lower()
        if "médecin" not in lower and "professionnel de santé" not in lower:
            raise ValueError("the disclaimer must tell to see a doctor")
        return value


# ==================== the chat with the assistant (POST /chat) ====================
class ChatMessage(BaseModel):
    # 1 message of the conversation, sent back by the page so the assistant knows what was said before
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    analysis: Optional[dict] = None          # the last result of /analyze_pcg kept by the page (None = no wav yet)
    history: List[ChatMessage] = []          # the last messages (only the last 6 are kept)

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, value):
        value = value.strip()
        if value == "":
            raise ValueError("the question is empty")
        return value

    @field_validator("history")
    @classmethod
    def keep_last_6(cls, value):
        return value[-6:]


class ChatAnswer(BaseModel):
    answer: str = Field(min_length=1)
    on_topic: bool                           # False = the question is not about the heart: polite refusal
    sources: List[Source] = []               # the RAG passages used for the answer
    disclaimer: str


def risk_level(probability):
    # the level of risk comes from the probability of "abnormal" (never from the LLM)
    if probability < 0.3:
        return "faible"
    if probability < 0.7:
        return "modéré"
    return "élevé"
