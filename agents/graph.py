"""
The graph of the 3 agents (LangGraph StateGraph), like in the project PDF:

    Analyste ──> Cardiologue RAG ──> Auditeur ──(approved)──────────────> end
                       ^                 │
                       └──(1 retry with the issues of the Auditeur)
                                         └──(still not approved, or no LLM)──> safe fallback report ──> end

  - Analyste   : reads the audio result (prediction, confidence, warnings, agreement between the 5 models),
                 writes a short technical analysis and the search query for the RAG
  - Cardiologue: searches the clinical documents in Qdrant (RAG) and explains the result, using ONLY these
                 passages, with citations [n]
  - Auditeur   : checks the report (Pydantic schema, coherent with the audio prediction, no definitive diagnosis,
                 no prescription, sources cited, a disclaimer that sends to a doctor) with rules + the LLM
Privacy: the LLM never gets the audio nor the file name, only the numbers of the audio result.
"""
import re
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from agents import llm, prompts
from agents.schemas import CardioReport, risk_level
from rag_engine.search import search as rag_search

LABELS = {"normal": "normal", "abnormal": "anormal"}    # the predictions, in French
MAX_RETRIES = 1                                         # the Cardiologue can correct its report once

# words that must never appear in a report: a definitive diagnosis or a prescription
FORBIDDEN = ["vous avez un ", "vous avez une ", "vous souffrez", "vous êtes atteint", "diagnostic confirmé",
             "posologie", "comprimé", " mg ", "milligramme", "aspirine", "bêta-bloquant", "beta-bloquant",
             "diurétique", "antibiotique", "anticoagulant"]

DEFAULT_DISCLAIMER = ("Ce rapport est produit par un outil d'aide au dépistage : seul un médecin peut poser un "
                      "diagnostic.")
EMERGENCY = ("En cas de douleur thoracique, de malaise ou d'essoufflement important, appelez immédiatement les "
             "urgences (SAMU : 190 en Tunisie, 15 ou 112 en France).")


class AgentState(TypedDict, total=False):
    audio: dict             # the result of the audio engine (checked by AudioResult before)
    facts: dict             # computed by the code from the audio result
    analysis: str           # written by the Analyste
    search_query: str       # written by the Analyste, used by the Cardiologue
    technical_chunks: list  # RAG "technique" passages (Analyste)
    chunks: list            # RAG "clinique" passages (Cardiologue)
    report: dict            # the report (CardioReport as a dict)
    issues: list            # the problems found by the Auditeur
    approved: bool
    retries: int
    fallback: bool
    llm_error: str
    metrics: dict           # latency of each agent, tokens, ... (for MLflow)


# ==================== helpers ====================
def compute_facts(audio):
    # the facts of the audio result, computed by the code (the LLM receives them, it does not compute them)
    n_models = len(audio["benchmark"])
    n_abnormal = 0
    n_agree = 0
    for answer in audio["benchmark"]:
        if answer["prediction"] == "abnormal":
            n_abnormal = n_abnormal + 1
        if answer["prediction"] == audio["prediction"]:
            n_agree = n_agree + 1
    probability = audio["probability_abnormal"]
    if probability is None:              # a model without probability (SVM): the share of models that say abnormal
        probability = n_abnormal / n_models
    return {
        "prediction": audio["prediction"],
        "label": LABELS[audio["prediction"]],
        "probability_abnormal": probability,
        "model": audio["model"],
        "risk_level": risk_level(probability),
        "n_models": n_models,
        "n_agree": n_agree,
        "warning": audio.get("warning"),
        "duration_s": audio["duration_s"],
    }


def facts_text(facts):
    if facts["warning"]:
        quality = facts["warning"]
    else:
        quality = "aucun problème signalé"
    return ("- prédiction du modèle principal (" + facts["model"] + ") : " + facts["label"] + "\n"
            "- probabilité d'anomalie : " + str(round(facts["probability_abnormal"], 2))
            + " -> niveau de risque : " + facts["risk_level"] + "\n"
            "- accord entre les modèles : " + str(facts["n_agree"]) + " sur " + str(facts["n_models"])
            + " donnent la même réponse\n"
            "- durée de l'enregistrement : " + str(facts["duration_s"]) + " s\n"
            "- qualité de l'enregistrement : " + quality)


def chunks_text(chunks):
    lines = []
    for i in range(len(chunks)):
        lines.append("[" + str(i + 1) + "] (" + chunks[i]["title"] + ", p. " + str(chunks[i]["page"]) + ") "
                     + chunks[i]["text"])
    return "\n".join(lines)


def safe_search(query, k, category):
    # the RAG search; if Qdrant is down, no passage (the Auditeur will see that no source is cited)
    try:
        return rag_search(query, k=k, category=category)
    except Exception:
        return []


def add_tokens(metrics, answer):
    metrics["prompt_tokens"] = metrics.get("prompt_tokens", 0) + answer["prompt_tokens"]
    metrics["eval_tokens"] = metrics.get("eval_tokens", 0) + answer["eval_tokens"]


def as_text(value):
    # the LLM should give a sentence; it sometimes gives a list of sentences -> 1 text
    if isinstance(value, list):
        return " ".join([str(item) for item in value])
    if value is None:
        return ""
    return str(value)


def as_text_list(value):
    # the LLM should give a list of sentences; accept a single sentence too
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip() != ""]
    if isinstance(value, str) and value.strip() != "":
        return [value]
    return []


# ==================== agent 1: Analyste ====================
def analyste(state):
    start = time.time()
    metrics = dict(state.get("metrics", {}))
    facts = compute_facts(state["audio"])

    # without the LLM, a default analysis and query made by the code
    analysis = ("Le modèle " + facts["model"] + " classe l'enregistrement comme " + facts["label"] + " (probabilité "
                + str(round(facts["probability_abnormal"], 2)) + "), " + str(facts["n_agree"]) + " modèles sur "
                + str(facts["n_models"]) + " sont d'accord.")
    if facts["prediction"] == "abnormal":
        search_query = "souffle cardiaque bruit surajouté auscultation valve examen"
    else:
        search_query = "bruits du cœur normaux B1 B2 auscultation"

    technical_chunks = safe_search("fiabilité de la classification automatique des sons cardiaques normaux anormaux",
                                   k=2, category="technique")
    llm_error = ""
    try:
        message = "Résultat du moteur audio :\n" + facts_text(facts)
        if technical_chunks:
            message = message + "\n\nExtraits d'articles techniques :\n" + chunks_text(technical_chunks)
        answer = llm.chat_json(prompts.ANALYSTE, message)
        add_tokens(metrics, answer)
        if as_text(answer["content"].get("analysis")).strip() != "":
            analysis = as_text(answer["content"]["analysis"])
        if as_text(answer["content"].get("search_query")).strip() != "":
            search_query = as_text(answer["content"]["search_query"])
    except Exception as error:
        llm_error = "LLM injoignable ou réponse invalide (Analyste) : " + str(error)[:200]

    metrics["analyste_s"] = round(time.time() - start, 2)
    return {"facts": facts, "analysis": analysis, "search_query": search_query, "technical_chunks": technical_chunks,
            "llm_error": llm_error, "metrics": metrics, "retries": 0, "issues": []}


# ==================== agent 2: Cardiologue RAG ====================
def cardiologue(state):
    start = time.time()
    metrics = dict(state.get("metrics", {}))
    facts = state["facts"]
    retries = state.get("retries", 0)
    if state.get("issues"):          # we come back from the Auditeur: this is the retry
        retries = retries + 1

    chunks = state.get("chunks")
    if not chunks:
        chunks = safe_search(state["search_query"], k=4, category="clinique")

    message = ("Résultat du moteur audio (faits calculés par le code) :\n" + facts_text(facts)
               + "\n\nAnalyse de l'agent Analyste : " + state["analysis"]
               + "\n\nExtraits des documents (à citer avec leur numéro) :\n" + chunks_text(chunks))
    if state.get("issues"):
        message = message + ("\n\nL'Auditeur a refusé ton rapport précédent pour ces raisons, corrige-les : "
                             + " ; ".join(state["issues"]))

    report = None
    llm_error = state.get("llm_error", "")
    try:
        answer = llm.chat_json(prompts.CARDIOLOGUE, message)
        add_tokens(metrics, answer)
        draft = answer["content"]

        # the sources = the passages cited by the LLM (their numbers), else all the passages it received
        sources = []
        for number in draft.get("sources_used", []):
            if isinstance(number, int) and 1 <= number <= len(chunks):
                chunk = chunks[number - 1]
                source = {"title": chunk["title"], "page": chunk["page"]}
                if source not in sources:
                    sources.append(source)
        if not sources:
            for chunk in chunks:
                source = {"title": chunk["title"], "page": chunk["page"]}
                if source not in sources:
                    sources.append(source)

        report = {
            # filled by the code, from the audio result
            "prediction": facts["prediction"],
            "probability_abnormal": round(facts["probability_abnormal"], 3),
            "model": facts["model"],
            "risk_level": facts["risk_level"],
            # written by the LLM
            "summary": as_text(draft.get("summary")),
            "clinical_explanation": as_text(draft.get("clinical_explanation")),
            "recommendations": as_text_list(draft.get("recommendations")),
            "sources": sources,
            "disclaimer": as_text(draft.get("disclaimer")),
            "audit": {"approved": False, "issues": []},    # filled by the Auditeur
        }
    except Exception as error:
        llm_error = "LLM injoignable ou réponse invalide (Cardiologue) : " + str(error)[:200]

    metrics["cardiologue_s"] = round(metrics.get("cardiologue_s", 0) + time.time() - start, 2)
    return {"chunks": chunks, "report": report, "retries": retries, "llm_error": llm_error, "metrics": metrics}


# ==================== agent 3: Auditeur ====================
def check_rules(report, facts):
    # the checks made by the code (always the same answer for the same report)
    issues = []
    if report is None:
        return ["pas de rapport (le LLM n'a pas répondu)"]
    try:
        CardioReport(**report)
    except ValidationError as error:
        for problem in error.errors():
            issues.append("schéma invalide : " + str(problem["loc"][0]) + " (" + problem["msg"] + ")")

    text = (report["summary"] + " " + report["clinical_explanation"] + " " + " ".join(report["recommendations"])).lower()
    if facts["prediction"] == "abnormal" and "anormal" not in report["summary"].lower():
        issues.append("le résumé ne dit pas que le son est classé « anormal »")
    if facts["prediction"] == "normal" and re.search(r"\bnormal", report["summary"].lower()) is None:
        issues.append("le résumé ne dit pas que le son est classé « normal »")
    if facts["prediction"] == "normal" and "anormal" in report["summary"].lower():
        issues.append("le résumé dit « anormal » alors que le moteur audio dit « normal »")
    for word in FORBIDDEN:
        if word in text:
            issues.append("diagnostic définitif ou prescription interdit : « " + word.strip() + " »")
    # the LLM must not say that all the models agree when they do not (a 3B model wrote "accord parfait" for 5 of 6)
    if facts["n_agree"] < facts["n_models"]:
        if re.search(r"accord[^.]{0,40}parfait|parfaitement d'accord|tous les modèles|unanim", text):
            issues.append("le rapport dit que tous les modèles sont d'accord, alors que seulement "
                          + str(facts["n_agree"]) + " sur " + str(facts["n_models"]) + " donnent la même réponse")
    if len(report["sources"]) == 0:
        issues.append("aucune source citée")
    doctor = False
    for sentence in report["recommendations"]:
        if "médecin" in sentence.lower() or "urgence" in sentence.lower() or "cardiologue" in sentence.lower():
            doctor = True
    if not doctor:
        issues.append("les recommandations ne disent pas de consulter un médecin")
    return issues


def is_true(value):
    # a small LLM writes true, "true", "oui"... -> True
    return value is True or str(value).strip().lower() in ["true", "oui", "yes", "vrai"]


def is_false(value):
    return value is False or str(value).strip().lower() in ["false", "non", "no", "faux"]


def llm_audit_issues(content):
    # the LLM answers 1 question per check (true / false) and the CODE decides. We do not ask a small LLM for
    # "approved + a list of problems": the 3B model copied the checklist into the list (even the checks that passed)
    # and refused good reports. The coherence with the audio prediction is checked by check_rules (the code).
    problem = as_text(content.get("probleme")).strip()
    issues = []
    if is_true(content.get("diagnostic_definitif")):
        issues.append("diagnostic définitif")
    if is_true(content.get("prescription")):
        issues.append("prescription d'un traitement")
    if is_false(content.get("consulter_medecin")):
        issues.append("ne recommande pas de consulter un médecin")
    if issues and problem != "":
        issues = [issue + " (auditeur : " + problem + ")" for issue in issues]
    return issues


def auditeur(state):
    start = time.time()
    metrics = dict(state.get("metrics", {}))
    report = state.get("report")
    issues = check_rules(report, state["facts"])

    # the rules are OK: the LLM checks the meaning (diagnosis, prescription, coherence)
    if not issues:
        try:
            message = ("Prédiction du moteur audio : " + state["facts"]["label"] + " (niveau de risque "
                       + state["facts"]["risk_level"] + ")\n\nRapport à vérifier :\nRésumé : " + report["summary"]
                       + "\nExplication : " + report["clinical_explanation"] + "\nRecommandations : "
                       + " | ".join(report["recommendations"]) + "\nAvertissement : " + report["disclaimer"])
            answer = llm.chat_json(prompts.AUDITEUR, message)
            add_tokens(metrics, answer)
            issues = issues + llm_audit_issues(answer["content"])
        except Exception:
            pass    # the LLM check is a second opinion: if it fails, the rules decide

    approved = len(issues) == 0
    if report is not None:
        report = dict(report)
        report["audit"] = {"approved": approved, "issues": issues}
    metrics["auditeur_s"] = round(metrics.get("auditeur_s", 0) + time.time() - start, 2)
    return {"report": report, "issues": issues, "approved": approved, "metrics": metrics}


def after_audit(state):
    # where to go after the Auditeur
    if state["approved"]:
        return "end"
    if state.get("retries", 0) < MAX_RETRIES and state.get("report") is not None:
        return "retry"
    return "fallback"


# ==================== the safe fallback report (written by the code, no LLM) ====================
def fallback(state):
    facts = state["facts"]
    if facts["prediction"] == "abnormal":
        summary = ("L'analyse automatique classe ce son cardiaque comme « anormal » (probabilité d'anomalie "
                   + str(round(facts["probability_abnormal"], 2)) + ", " + str(facts["n_agree"]) + " modèles sur "
                   + str(facts["n_models"]) + " sont d'accord).")
        explanation = ("Un son classé anormal peut correspondre à un souffle ou à un bruit surajouté, mais il peut "
                       "aussi venir d'un enregistrement de mauvaise qualité. Ce n'est pas un diagnostic : une "
                       "auscultation par un médecin est nécessaire.")
        recommendations = ["Consultez un médecin pour une auscultation et, si besoin, une échocardiographie.",
                           EMERGENCY]
    else:
        summary = ("L'analyse automatique classe ce son cardiaque comme « normal » (probabilité d'anomalie "
                   + str(round(facts["probability_abnormal"], 2)) + ", " + str(facts["n_agree"]) + " modèles sur "
                   + str(facts["n_models"]) + " sont d'accord).")
        explanation = ("Le son ressemble aux enregistrements normaux (bruits B1 et B2 sans souffle marqué). Un "
                       "résultat normal n'exclut pas une maladie cardiaque.")
        recommendations = ["Consultez un médecin si vous avez des symptômes (essoufflement, douleur thoracique, "
                           "palpitations), même avec un résultat normal.", EMERGENCY]
    if facts["warning"]:
        explanation = explanation + " Attention : " + facts["warning"] + "."

    sources = []
    for chunk in state.get("chunks") or []:
        source = {"title": chunk["title"], "page": chunk["page"]}
        if source not in sources:
            sources.append(source)
    issues = list(state.get("issues") or [])
    if state.get("llm_error"):
        issues.append(state["llm_error"])
    issues.append("rapport de secours écrit par le code : le texte du LLM n'a pas été validé")

    report = {
        "prediction": facts["prediction"],
        "probability_abnormal": round(facts["probability_abnormal"], 3),
        "model": facts["model"],
        "risk_level": facts["risk_level"],
        "summary": summary,
        "clinical_explanation": explanation,
        "recommendations": recommendations,
        "sources": sources,
        "disclaimer": DEFAULT_DISCLAIMER,
        "audit": {"approved": False, "issues": issues},
    }
    return {"report": report, "fallback": True}


# ==================== the graph ====================
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("analyste", analyste)
    graph.add_node("cardiologue", cardiologue)
    graph.add_node("auditeur", auditeur)
    graph.add_node("fallback", fallback)

    graph.set_entry_point("analyste")
    graph.add_edge("analyste", "cardiologue")
    graph.add_edge("cardiologue", "auditeur")
    graph.add_conditional_edges("auditeur", after_audit, {"end": END, "retry": "cardiologue", "fallback": "fallback"})
    graph.add_edge("fallback", END)
    return graph.compile()


agents_graph = build_graph()


def run_agents(audio):
    # audio = the audio engine result (a dict already checked by AudioResult) -> (the report, the metrics)
    start = time.time()
    audio = dict(audio)
    audio.pop("filename", None)          # privacy: the file name never goes to the LLM
    state = agents_graph.invoke({"audio": audio, "metrics": {}})

    report = CardioReport(**state["report"]).model_dump()    # the final guardrail: a valid report or an error
    metrics = dict(state["metrics"])
    metrics["total_s"] = round(time.time() - start, 2)
    metrics["approved"] = state["report"]["audit"]["approved"]
    metrics["retries"] = state.get("retries", 0)
    metrics["fallback"] = state.get("fallback", False)
    metrics["schema_valid"] = True
    metrics["n_sources"] = len(report["sources"])
    metrics["risk_level"] = report["risk_level"]
    return report, metrics
