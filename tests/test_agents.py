"""
Compliance tests of the agents (Service 4): the Pydantic schema of the report (valid / invalid), the rules of the
Auditeur, and the LangGraph graph with a FAKE LLM and a FAKE RAG (no Ollama, no Qdrant needed):
happy path, retry after the Auditeur, safe fallback, LLM down, privacy (no file name sent to the LLM).
"""
import pytest
import requests
from fastapi.testclient import TestClient
from pydantic import ValidationError

import agents.graph as graph
import agents.llm as llm
import agents.main as orchestrator
from agents import prompts
from agents.schemas import AudioResult, CardioReport, risk_level

CHUNKS = [
    {"text": "Un souffle systolique s'entend entre B1 et B2.", "source": "a.pdf", "title": "Cours A", "page": 3,
     "category": "clinique", "score": 0.8},
    {"text": "L'échocardiographie confirme une valvulopathie.", "source": "b.pdf", "title": "Cours B", "page": 7,
     "category": "clinique", "score": 0.7},
]
GOOD_REPORT_TEXT = {
    "summary": "L'analyse classe ce son cardiaque comme anormal, avec un niveau de risque élevé.",
    "clinical_explanation": "Un son anormal peut correspondre à un souffle entre B1 et B2 [1]. Seul un examen "
                            "comme l'échocardiographie permet de conclure [2].",
    "recommendations": ["Consultez un médecin rapidement pour une auscultation.",
                        "En cas de douleur thoracique, appelez les urgences."],
    "disclaimer": "Seul un médecin peut poser un diagnostic.",
    "sources_used": [1, 2],
}


def valid_report():
    return {
        "prediction": "abnormal", "probability_abnormal": 0.86, "model": "CNN", "risk_level": "élevé",
        "summary": GOOD_REPORT_TEXT["summary"], "clinical_explanation": GOOD_REPORT_TEXT["clinical_explanation"],
        "recommendations": GOOD_REPORT_TEXT["recommendations"], "sources": [{"title": "Cours A", "page": 3}],
        "disclaimer": GOOD_REPORT_TEXT["disclaimer"], "audit": {"approved": True, "issues": []},
    }


# the answer of the Auditeur LLM for a good report (1 true / false per check)
AUDIT_OK = {"diagnostic_definitif": False, "prescription": False, "consulter_medecin": True, "probleme": ""}


class FakeLLM:
    # replaces llm.chat_json: gives a prepared answer for each agent and remembers the prompts it received
    def __init__(self, cardiologue_answers=None, auditeur_answers=None):
        self.cardiologue_answers = cardiologue_answers or [GOOD_REPORT_TEXT]
        self.auditeur_answers = auditeur_answers or [AUDIT_OK]
        self.prompts = []
        self.calls = {"analyste": 0, "cardiologue": 0, "auditeur": 0}

    def __call__(self, system_prompt, user_prompt):
        self.prompts.append(user_prompt)
        if system_prompt == prompts.ANALYSTE:
            name = "analyste"
            content = {"analysis": "4 modèles sur 5 disent anormal.", "search_query": "souffle cardiaque"}
        elif system_prompt == prompts.CARDIOLOGUE:
            name = "cardiologue"
            content = self.cardiologue_answers[min(self.calls[name], len(self.cardiologue_answers) - 1)]
        else:
            name = "auditeur"
            content = self.auditeur_answers[min(self.calls[name], len(self.auditeur_answers) - 1)]
        self.calls[name] = self.calls[name] + 1
        return {"content": content, "prompt_tokens": 100, "eval_tokens": 50, "seconds": 0.1}


@pytest.fixture
def fake_rag(monkeypatch):
    monkeypatch.setattr(graph, "rag_search", lambda question, k=4, category=None: CHUNKS[:k])


# ==================== Pydantic schemas ====================
def test_valid_report_is_accepted():
    report = CardioReport(**valid_report())
    assert report.risk_level == "élevé"


@pytest.mark.parametrize("field, value", [
    ("prediction", "maybe"),                                  # only "normal" or "abnormal"
    ("risk_level", "très élevé"),                             # only "faible", "modéré" or "élevé"
    ("recommendations", []),                                  # at least 1 recommendation
    ("disclaimer", "Ceci est un outil."),                     # must send to a doctor
    ("probability_abnormal", 1.7),                            # a probability is between 0 and 1
    ("summary", "court"),                                     # too short to be a summary
])
def test_invalid_report_is_refused(field, value):
    report = valid_report()
    report[field] = value
    with pytest.raises(ValidationError):
        CardioReport(**report)


def test_report_without_audit_is_refused():
    report = valid_report()
    del report["audit"]
    with pytest.raises(ValidationError):
        CardioReport(**report)


def test_audio_result_schema(audio_result):
    assert AudioResult(**audio_result).model == "CNN"
    audio_result["prediction"] = "unknown"
    with pytest.raises(ValidationError):
        AudioResult(**audio_result)


def test_risk_level_thresholds():
    assert risk_level(0.1) == "faible"
    assert risk_level(0.5) == "modéré"
    assert risk_level(0.95) == "élevé"


# ==================== facts and rules (code, no LLM) ====================
def test_compute_facts(audio_result):
    facts = graph.compute_facts(audio_result)
    assert facts["n_agree"] == 4 and facts["n_models"] == 5
    assert facts["risk_level"] == "élevé" and facts["label"] == "anormal"


def test_compute_facts_without_probability(audio_result):
    audio_result["probability_abnormal"] = None                # e.g. the SVM is the official model
    facts = graph.compute_facts(audio_result)
    assert facts["probability_abnormal"] == pytest.approx(4 / 5)    # the share of models that say abnormal


def test_rules_accept_a_good_report(audio_result):
    assert graph.check_rules(valid_report(), graph.compute_facts(audio_result)) == []


def test_rules_find_diagnosis_prescription_and_missing_sources(audio_result):
    report = valid_report()
    report["clinical_explanation"] = "Vous avez une insuffisance mitrale, prenez de l'aspirine chaque matin."
    report["sources"] = []
    issues = graph.check_rules(report, graph.compute_facts(audio_result))
    assert any("vous avez une" in issue for issue in issues)
    assert any("aspirine" in issue for issue in issues)
    assert "aucune source citée" in issues


def test_rules_find_a_false_agreement_of_the_models(audio_result):
    # in audio_result, 4 of the 5 models say abnormal: "all the models agree" is false
    report = valid_report()
    report["summary"] = "Le son est classé anormal : l'accord entre les modèles est parfait."
    issues = graph.check_rules(report, graph.compute_facts(audio_result))
    assert any("4 sur 5" in issue for issue in issues)
    report["summary"] = "Le son est classé anormal : l'outil n'est pas parfait, 4 modèles sur 5 sont d'accord."
    assert graph.check_rules(report, graph.compute_facts(audio_result)) == []


def test_rules_find_a_summary_that_contradicts_the_audio(audio_result):
    report = valid_report()
    report["summary"] = "Le son de votre cœur est tout à fait normal et rassurant."
    issues = graph.check_rules(report, graph.compute_facts(audio_result))
    assert any("anormal" in issue for issue in issues)


def test_rules_need_a_doctor_in_the_recommendations(audio_result):
    report = valid_report()
    report["recommendations"] = ["Buvez de l'eau."]
    issues = graph.check_rules(report, graph.compute_facts(audio_result))
    assert any("médecin" in issue for issue in issues)


# ==================== the answer of the Auditeur LLM -> the issues (the code decides) ====================
def test_auditor_llm_approves_a_good_report():
    assert graph.llm_audit_issues(AUDIT_OK) == []


def test_auditor_llm_finds_each_problem():
    bad = {"diagnostic_definitif": True, "prescription": "true", "consulter_medecin": False,
           "probleme": "le rapport affirme une sténose"}
    issues = graph.llm_audit_issues(bad)
    assert len(issues) == 3 and all("sténose" in issue for issue in issues)


def test_auditor_llm_checklist_copied_in_the_text_is_not_a_problem():
    # the bug seen with the 3B model: the checks that PASSED were written as problems -> only the true/false count
    answer = dict(AUDIT_OK, probleme="aucun diagnostic définitif ; aucune prescription")
    assert graph.llm_audit_issues(answer) == []


# ==================== the graph with a fake LLM ====================
def test_graph_happy_path(monkeypatch, fake_rag, audio_result):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "chat_json", fake)
    report, metrics = graph.run_agents(audio_result)
    assert report["audit"]["approved"] is True and metrics["retries"] == 0 and metrics["fallback"] is False
    # the fields of the audio result are filled by the code, not by the LLM
    assert report["prediction"] == "abnormal" and report["model"] == "CNN" and report["risk_level"] == "élevé"
    assert report["sources"] == [{"title": "Cours A", "page": 3}, {"title": "Cours B", "page": 7}]
    assert fake.calls == {"analyste": 1, "cardiologue": 1, "auditeur": 1}
    assert metrics["prompt_tokens"] == 300 and metrics["schema_valid"] is True


def test_graph_retries_once_after_the_auditor(monkeypatch, fake_rag, audio_result):
    fake = FakeLLM(auditeur_answers=[dict(AUDIT_OK, diagnostic_definitif=True, probleme="trop affirmatif"),
                                     AUDIT_OK])
    monkeypatch.setattr(llm, "chat_json", fake)
    report, metrics = graph.run_agents(audio_result)
    assert metrics["retries"] == 1 and report["audit"]["approved"] is True
    assert fake.calls["cardiologue"] == 2
    assert "trop affirmatif" in fake.prompts[-2]               # the Cardiologue received the issues


def test_graph_falls_back_to_a_safe_report(monkeypatch, fake_rag, audio_result):
    bad = dict(GOOD_REPORT_TEXT)
    bad["clinical_explanation"] = "Vous avez une valvulopathie certaine : prenez un bêta-bloquant 50 mg par jour."
    fake = FakeLLM(cardiologue_answers=[bad])
    monkeypatch.setattr(llm, "chat_json", fake)
    report, metrics = graph.run_agents(audio_result)
    assert metrics["fallback"] is True and metrics["retries"] == 1
    assert report["audit"]["approved"] is False
    assert "bêta-bloquant" not in report["clinical_explanation"]        # the bad text is never shown
    assert "anormal" in report["summary"] and "médecin" in report["disclaimer"]
    CardioReport(**report)                                             # the fallback report is valid


def test_graph_works_when_the_llm_is_down(monkeypatch, fake_rag, audio_result):
    def llm_down(system_prompt, user_prompt):
        raise requests.ConnectionError("Ollama is down")

    monkeypatch.setattr(llm, "chat_json", llm_down)
    report, metrics = graph.run_agents(audio_result)
    assert metrics["fallback"] is True
    assert any("LLM injoignable" in issue for issue in report["audit"]["issues"])
    assert report["sources"] != []                                     # the RAG passages are still cited


def test_graph_works_for_a_normal_sound(monkeypatch, fake_rag, audio_result):
    audio_result["prediction"] = "normal"
    audio_result["probability_abnormal"] = 0.1
    normal_text = dict(GOOD_REPORT_TEXT)
    normal_text["summary"] = "L'analyse classe ce son cardiaque comme normal, avec un risque faible."
    monkeypatch.setattr(llm, "chat_json", FakeLLM(cardiologue_answers=[normal_text]))
    report, metrics = graph.run_agents(audio_result)
    assert report["risk_level"] == "faible" and report["audit"]["approved"] is True


def test_the_file_name_never_goes_to_the_llm(monkeypatch, fake_rag, audio_result):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "chat_json", fake)
    graph.run_agents(audio_result)
    for prompt in fake.prompts:
        assert "patient_042" not in prompt


# ==================== the LLM wrapper (fake Ollama) ====================
class FakeOllamaResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_chat_json_reads_the_ollama_answer(monkeypatch):
    sent = {}

    def fake_post(url, json=None, timeout=None):
        sent.update(json)
        return FakeOllamaResponse({"message": {"content": '{"approved": true}'}, "prompt_eval_count": 12,
                                   "eval_count": 3, "total_duration": 2_000_000_000})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    answer = llm.chat_json("system", "user")
    assert answer == {"content": {"approved": True}, "prompt_tokens": 12, "eval_tokens": 3, "seconds": 2.0}
    assert sent["format"] == "json" and sent["options"]["temperature"] == 0.2


def test_chat_json_refuses_an_answer_that_is_not_json(monkeypatch):
    monkeypatch.setattr(llm.requests, "post",
                        lambda url, json=None, timeout=None: FakeOllamaResponse({"message": {"content": "hello"}}))
    with pytest.raises(ValueError):
        llm.chat_json("system", "user")


def test_is_ready(monkeypatch):
    monkeypatch.setattr(llm.requests, "get",
                        lambda url, timeout=None: FakeOllamaResponse({"models": [{"name": llm.LLM_MODEL}]}))
    assert llm.is_ready() is True

    def down(url, timeout=None):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(llm.requests, "get", down)
    assert llm.is_ready() is False


# ==================== the orchestrator API ====================
def test_orchestrate_endpoint(monkeypatch, fake_rag, audio_result):
    monkeypatch.setattr(llm, "chat_json", FakeLLM())
    monkeypatch.setattr(orchestrator, "MLFLOW_TRACKING_URI", "")
    client = TestClient(orchestrator.app)
    response = client.post("/orchestrate", json=audio_result)
    assert response.status_code == 200
    data = response.json()
    assert data["report"]["audit"]["approved"] is True
    assert data["audio"]["prediction"] == "abnormal"


def test_orchestrate_refuses_a_wrong_audio_result(audio_result):
    client = TestClient(orchestrator.app)
    audio_result["benchmark"] = []                              # at least 1 model is needed
    assert client.post("/orchestrate", json=audio_result).status_code == 422


def test_orchestrator_health(monkeypatch):
    def qdrant_down(url, timeout=None):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(orchestrator.requests, "get", qdrant_down)
    monkeypatch.setattr(llm, "is_ready", lambda: True)
    data = TestClient(orchestrator.app).get("/health").json()
    assert data["qdrant_ready"] is False and data["llm_ready"] is True


def test_llm_inference_logging_never_breaks(monkeypatch):
    monkeypatch.setattr(orchestrator, "MLFLOW_TRACKING_URI", "")
    orchestrator.log_llm_inference({"total_s": 1.0})
    monkeypatch.setattr(orchestrator, "MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    orchestrator.log_llm_inference({"total_s": 1.0, "approved": True})    # MLflow unreachable: no exception
