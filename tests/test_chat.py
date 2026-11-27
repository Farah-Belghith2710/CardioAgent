"""
Tests of the chat with the assistant (agents/chat.py, POST /chat of the orchestrator and of the gateway), with a FAKE
LLM and a FAKE RAG: the off-topic guardrail, the answers about the analysis, the safety checks, the schemas.
"""
import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import ValidationError

import agents.main as orchestrator
import api.main as gateway
from agents import chat, llm, prompts
from agents.schemas import ChatAnswer, ChatRequest

CHUNKS = [
    {"text": "Un souffle est un bruit dû à un flux sanguin turbulent.", "title": "Cours A", "page": 3,
     "source": "a.pdf", "category": "clinique", "score": 0.8},
    {"text": "Les bruits B1 et B2 sont les bruits normaux du cœur.", "title": "Cours B", "page": 7,
     "source": "b.pdf", "category": "clinique", "score": 0.7},
]


class FakeLLM:
    # replaces llm.chat_json: an answer for the guardrail and one for the chat, and it counts the calls
    def __init__(self, answer=None, on_topic=True):
        self.answer = answer or {"answer": "Le résultat est anormal [1]. Parlez-en à un médecin.", "sources_used": [1]}
        self.on_topic = on_topic
        self.calls = {"guard": 0, "chat": 0}
        self.prompts = []

    def __call__(self, system_prompt, user_prompt, max_tokens=None):
        self.prompts.append(user_prompt)
        if system_prompt == prompts.CHAT_GUARD:
            self.calls["guard"] = self.calls["guard"] + 1
            content = {"on_topic": self.on_topic}
        else:
            self.calls["chat"] = self.calls["chat"] + 1
            content = self.answer
        return {"content": content, "prompt_tokens": 80, "eval_tokens": 40, "seconds": 0.1}


@pytest.fixture
def fake_rag(monkeypatch):
    searches = []

    def search(question, k=4, category=None):
        searches.append((question, k, category))
        return CHUNKS[:k]

    monkeypatch.setattr(chat, "safe_search", search)
    return searches


@pytest.fixture
def analysis(audio_result):
    # what the page sends: the last answer of /analyze_pcg (audio result + report of the agents)
    return {"audio": audio_result, "report": {"summary": "Le son est classé anormal par le CNN."}}


# ==================== the guardrail ====================
@pytest.mark.parametrize("question", ["Comment faire une pizza ?", "recette de pizza", "Give me a cupcake recipe",
                                      "Qui va gagner le match de football ?"])
def test_off_topic_question_gets_the_fixed_refusal(monkeypatch, fake_rag, question, analysis):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "chat_json", fake)
    answer, metrics = chat.chat(question, analysis)
    assert answer["on_topic"] is False and answer["sources"] == []
    assert answer["answer"] in chat.REFUSAL.values()                 # the text written by the code
    assert fake.calls == {"guard": 0, "chat": 0} and fake_rag == []  # no LLM, no RAG search
    assert metrics["guardrail"] == "keyword"


def test_refusal_is_in_the_language_of_the_question(monkeypatch, fake_rag):
    monkeypatch.setattr(llm, "chat_json", FakeLLM())
    assert chat.chat("How do I make a cupcake recipe?")[0]["answer"] == chat.REFUSAL["en"]
    assert chat.chat("Comment faire une pizza ?")[0]["answer"] == chat.REFUSAL["fr"]
    assert chat.chat("Give me a cupcake recipe")[0]["answer"] == chat.REFUSAL["en"]


def test_unclear_question_is_decided_by_the_llm(monkeypatch, fake_rag):
    fake = FakeLLM(on_topic=False)
    monkeypatch.setattr(llm, "chat_json", fake)
    answer, metrics = chat.chat("Quelle est la capitale de l'Australie ?")
    assert answer["on_topic"] is False and fake.calls == {"guard": 1, "chat": 0}
    assert metrics["guardrail"] == "llm"


def test_unclear_question_and_llm_down_gives_the_refusal(monkeypatch, fake_rag):
    def llm_down(system_prompt, user_prompt, max_tokens=None):
        raise requests.ConnectionError("Ollama is down")

    monkeypatch.setattr(llm, "chat_json", llm_down)
    answer, _ = chat.chat("Quelle est la capitale de l'Australie ?")
    assert answer["on_topic"] is False


def test_keywords():
    assert chat.is_on_topic("C'est quoi un souffle cardiaque ?") == (True, "keyword")
    assert chat.is_on_topic("Dois-je voir un MÉDECIN ?") == (True, "keyword")
    assert chat.is_on_topic("Que veut dire 0.86 ?", has_analysis=True) == (True, "keyword")
    assert chat.is_on_topic("Pourquoi les modèles ne sont pas d'accord ?", has_analysis=True) == (True, "keyword")
    assert chat.is_on_topic("Est-ce que la pizza est mauvaise pour le cœur ?")[0] is True    # about the heart
    assert chat.language("Should I see a doctor?") == "en" and chat.language("Dois-je voir un médecin ?") == "fr"


# ==================== the answers ====================
def test_on_topic_question_with_an_abnormal_analysis(monkeypatch, fake_rag, analysis):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "chat_json", fake)
    answer, metrics = chat.chat("Dois-je voir un médecin ?", analysis)
    assert answer["on_topic"] is True
    assert "médecin" in answer["answer"] and answer["sources"] == [{"title": "Cours A", "page": 3}]
    assert fake_rag == [("Dois-je voir un médecin ?", 4, "clinique")]
    assert "0.86" in fake.prompts[-1] and "CNN" in fake.prompts[-1]    # the numbers of the analysis go to the LLM
    assert metrics["n_sources"] == 1 and metrics["safe_answer"] is False
    ChatAnswer(**answer)


def test_abnormal_result_always_sends_to_a_cardiologist(monkeypatch, fake_rag, analysis):
    monkeypatch.setattr(llm, "chat_json", FakeLLM(answer={"answer": "Ce son est classé anormal [2].", "sources_used": [2]}))
    answer, _ = chat.chat("Ce son est-il normal ?", analysis)
    assert "cardiologue" in answer["answer"]
    assert answer["answer"].startswith("Ce son est classé anormal [1].")       # [2] renumbered as the 1st source
    assert answer["sources"] == [{"title": "Cours B", "page": 7}]


def test_my_sound_without_analysis_asks_to_upload(monkeypatch, fake_rag):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "chat_json", fake)
    answer, _ = chat.chat("Est-ce que mon son est normal ?")
    assert answer["answer"] == chat.UPLOAD_FIRST["fr"] and answer["on_topic"] is True
    assert fake.calls["chat"] == 0


def test_general_heart_question_without_analysis(monkeypatch, fake_rag):
    fake = FakeLLM(answer={"answer": "Un souffle est un bruit dû à un flux turbulent [1].", "sources_used": [1]})
    monkeypatch.setattr(llm, "chat_json", fake)
    answer, _ = chat.chat("C'est quoi un souffle cardiaque ?")
    assert "souffle" in answer["answer"] and answer["sources"] != []
    assert "Aucun enregistrement" in fake.prompts[-1]


@pytest.mark.parametrize("bad_answer", ["Vous avez une sténose aortique.", "Prenez de l'aspirine 100 mg.",
                                        "La probabilité est de 0.42, c'est rassurant."])
def test_diagnosis_prescription_or_invented_number_is_replaced(monkeypatch, fake_rag, analysis, bad_answer):
    monkeypatch.setattr(llm, "chat_json", FakeLLM(answer={"answer": bad_answer, "sources_used": []}))
    answer, metrics = chat.chat("Ce son est-il grave ?", analysis)
    assert metrics["safe_answer"] is True
    assert bad_answer not in answer["answer"] and "0.86" in answer["answer"] and "médecin" in answer["answer"]


def test_numbers_of_the_analysis_are_allowed(monkeypatch, fake_rag, analysis):
    good = {"answer": "La probabilité d'anomalie est de 0,86 (86 %). Parlez-en à un médecin.", "sources_used": []}
    monkeypatch.setattr(llm, "chat_json", FakeLLM(answer=good))
    answer, metrics = chat.chat("Que signifie la probabilité ?", analysis)
    assert metrics["safe_answer"] is False and "0,86" in answer["answer"]


def test_emergency_symptoms_give_the_emergency_number(monkeypatch, fake_rag, analysis):
    monkeypatch.setattr(llm, "chat_json", FakeLLM())
    answer, _ = chat.chat("J'ai une douleur thoracique, que faire ?", analysis)
    assert "190" in answer["answer"]


def test_llm_down_gives_a_safe_answer(monkeypatch, fake_rag, analysis):
    def llm_down(system_prompt, user_prompt, max_tokens=None):
        raise requests.ConnectionError("Ollama is down")

    monkeypatch.setattr(llm, "chat_json", llm_down)
    answer, metrics = chat.chat("Dois-je consulter un médecin ?", analysis)
    assert metrics["safe_answer"] is True and "anormal" in answer["answer"] and "médecin" in answer["answer"]


def test_history_goes_to_the_llm(monkeypatch, fake_rag, analysis):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "chat_json", fake)
    history = [{"role": "user", "content": "Ce son est-il normal ?"}, {"role": "assistant", "content": "Il est classé anormal."}]
    chat.chat("Et que dois-je faire ?", analysis, history)
    assert "Il est classé anormal." in fake.prompts[-1]


# ==================== the schemas ====================
def test_chat_request_schema():
    request = ChatRequest(question="  Dois-je voir un médecin ?  ", history=[{"role": "user", "content": str(i)} for i in range(10)])
    assert request.question == "Dois-je voir un médecin ?" and len(request.history) == 6    # only the last 6
    for bad in [{"question": ""}, {"question": "   "}, {"question": "x" * 1001},
                {"question": "ok", "history": [{"role": "robot", "content": "hi"}]}]:
        with pytest.raises(ValidationError):
            ChatRequest(**bad)


def test_chat_answer_schema():
    ChatAnswer(answer="Bonjour", on_topic=True, sources=[{"title": "A", "page": 1}], disclaimer="Voir un médecin.")
    with pytest.raises(ValidationError):
        ChatAnswer(answer="", on_topic=True, disclaimer="x")


# ==================== the routes ====================
def test_orchestrator_chat_route(monkeypatch, fake_rag, analysis):
    monkeypatch.setattr(llm, "chat_json", FakeLLM())
    monkeypatch.setattr(orchestrator, "MLFLOW_TRACKING_URI", "")
    client = TestClient(orchestrator.app)
    response = client.post("/chat", json={"question": "Dois-je voir un médecin ?", "analysis": analysis})
    assert response.status_code == 200 and response.json()["on_topic"] is True
    assert client.post("/chat", json={"question": ""}).status_code == 422


class FakeResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data
        self.text = str(data)

    def json(self):
        return self._data


def test_gateway_forwards_the_chat(monkeypatch):
    monkeypatch.setattr(gateway, "ORCHESTRATOR_URL", "http://agent-orchestrator:8002/orchestrate")
    sent = {}

    def fake_post(url, json=None, timeout=None, **kwargs):
        sent["url"] = url
        sent["json"] = json
        return FakeResponse(200, {"answer": "ok", "on_topic": True, "sources": [], "disclaimer": "médecin"})

    monkeypatch.setattr(gateway.requests, "post", fake_post)
    response = TestClient(gateway.app).post("/chat", json={"question": "Bonjour"})
    assert response.status_code == 200 and response.json()["answer"] == "ok"
    assert sent["url"] == "http://agent-orchestrator:8002/chat" and sent["json"] == {"question": "Bonjour"}


def test_gateway_chat_when_the_orchestrator_is_down(monkeypatch):
    monkeypatch.setattr(gateway, "ORCHESTRATOR_URL", "http://agent-orchestrator:8002/orchestrate")

    def down(url, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(gateway.requests, "post", down)
    assert TestClient(gateway.app).post("/chat", json={"question": "Bonjour"}).status_code == 503
    monkeypatch.setattr(gateway, "ORCHESTRATOR_URL", "")
    assert TestClient(gateway.app).post("/chat", json={"question": "Bonjour"}).status_code == 503


def test_gateway_passes_the_422_of_the_orchestrator(monkeypatch):
    monkeypatch.setattr(gateway, "ORCHESTRATOR_URL", "http://agent-orchestrator:8002/orchestrate")
    monkeypatch.setattr(gateway.requests, "post", lambda url, **kwargs: FakeResponse(422, {"detail": "question vide"}))
    response = TestClient(gateway.app).post("/chat", json={"question": ""})
    assert response.status_code == 422 and response.json()["detail"] == "question vide"


def test_gateway_audio_only_then_report(monkeypatch, heart_wav, audio_result):
    # the web page: 1. /analyze_pcg?agents=false (the audio result only), 2. /report with it
    monkeypatch.setattr(gateway, "AUDIO_ENGINE_URL", "http://audio-engine:8001/predict")
    monkeypatch.setattr(gateway, "ORCHESTRATOR_URL", "http://agent-orchestrator:8002/orchestrate")
    urls = []

    def fake_post(url, **kwargs):
        urls.append(url)
        if url.endswith("/predict"):
            return FakeResponse(200, audio_result)
        return FakeResponse(200, {"audio": audio_result, "report": {"summary": "..."}})

    monkeypatch.setattr(gateway.requests, "post", fake_post)
    client = TestClient(gateway.app)
    response = client.post("/analyze_pcg?agents=false", files={"file": ("a.wav", heart_wav, "audio/wav")})
    assert response.json() == audio_result and urls == ["http://audio-engine:8001/predict"]
    response = client.post("/report", json=audio_result)
    assert response.json()["report"] == {"summary": "..."} and urls[-1].endswith("/orchestrate")
    monkeypatch.setattr(gateway, "ORCHESTRATOR_URL", "")
    assert client.post("/report", json=audio_result).status_code == 503
