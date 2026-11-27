"""
The chat with the assistant (Service 4, POST /chat): questions about the analysed wav and about the heart.

    question ──> 1. guardrail: is it about the heart / the recording?
                    no ──> polite refusal written by the CODE (no RAG search, no answer written by the LLM)
             ──> 2. about "my sound" but no wav analysed yet ──> ask to upload a .wav first
             ──> 3. RAG search (clinical documents, k = 4) + the facts of the analysis ──> LLM (short JSON answer)
             ──> 4. safety checks by the code: no diagnosis or prescription, no invented number, a doctor for an
                    abnormal result, the emergency number for chest pain / fainting / strong shortness of breath

The numbers (prediction, probability, the answers of the models) come from the analysis JSON sent by the page:
the code gives them to the LLM and refuses an answer with a number that is not in the analysis.
"""
import re
import time
import unicodedata

from agents import llm, prompts
from agents.graph import EMERGENCY, FORBIDDEN, LABELS, chunks_text, compute_facts, facts_text, safe_search
from agents.schemas import AudioResult

MAX_TOKENS = 300    # short answers: the LLM runs on a CPU

# ---------- 1. the guardrail: lists of word beginnings (the text is in lower case, without accents) ----------
HEART_WORDS = ["coeur", "cardi", "souffle", "murmur", "heart", "auscult", "stethoscop", "valv", "palpitation",
               "tachycard", "bradycard", "arythm", "arrhythm", "systol", "diastol", "b1", "b2", "b3", "b4", "pcg",
               "phonocardio", "ecg", "electrocardio", "echocardio", "medecin", "docteur", "doctor", "urgence",
               "emergency", "samu", "poitrine", "thorac", "chest", "essouffl", "breath", "malaise", "evanou",
               "faint", "pouls", "pulse", "battement", "infarctus", "stenose", "valvul", "symptom"]
ANALYSIS_WORDS = ["son ", "sons ", "sound", "enregistr", "record", "wav", "fichier", "file", "resultat", "result",
                  "probabilit", "modele", "model", "analyse", "analysis", "normal", "anormal", "abnormal", "risque",
                  "risk", "fiab", "reliab", "ensemble", "cnn", "svm", "resnet", "transformer", "chiffre", "number",
                  "pourcentage", "percent", "score", "recall", "precision", "f1", "prediction", "predict",
                  "signifi", "veut dire", "mean", "accord", "agree"]
OFF_TOPIC_WORDS = ["pizza", "recette", "recipe", "cupcake", "gateau", "cake", "cuisin", "cook", "patisserie",
                   "football", "match", "politi", "president", "election", "devoir", "homework", "math",
                   "code", "python", "javascript", "programm", "film", "movie", "musique", "music", "meteo",
                   "weather", "voyage", "travel", "jeu video", "video game"]
GREETINGS = ["bonjour", "bonsoir", "salut", "merci", "hello", "hi", "thanks", "thank you", "coucou"]
# questions about "my" recording: they need an analysed wav
MY_RECORDING = ["mon son", "mon enregistr", "mon resultat", "mon analyse", "mon fichier", "mon coeur", "ce son",
                "ce resultat", "cette analyse", "cet enregistr", "la probabilit", "les modeles", "my sound",
                "my record", "my result", "my heart", "this sound", "the result", "the probabilit", "the models"]

# ---------- the texts written by the code (never by the LLM) ----------
REFUSAL = {
    "fr": ("Désolé, je suis un assistant spécialisé dans l'analyse des sons du cœur. Je ne peux pas répondre à cette "
           "question, mais je peux vous aider à comprendre votre enregistrement ou répondre à une question sur le "
           "cœur."),
    "en": ("Sorry, I am an assistant specialised in heart-sound analysis. I can't answer this question, but I can "
           "help you understand your recording or answer a question about the heart."),
}
UPLOAD_FIRST = {
    "fr": ("Je n'ai pas encore d'enregistrement à analyser. Envoyez d'abord un fichier .wav (à gauche), puis posez "
           "à nouveau votre question. Je peux déjà répondre aux questions générales sur le cœur."),
    "en": ("I don't have a recording yet. Please upload a .wav file first (on the left), then ask your question "
           "again. I can already answer general questions about the heart."),
}
DISCLAIMER = {
    "fr": "Cet assistant ne remplace pas un médecin : seul un médecin peut confirmer un résultat.",
    "en": "This assistant does not replace a doctor: only a doctor can confirm a result.",
}
ONLINE_DOWN = {
    "fr": "Le service d'IA en ligne n'est pas disponible pour le moment (vérifiez la clé dans .env).",
    "en": "The online AI service is not available right now (check the key in .env).",
}
SEE_CARDIOLOGIST = {
    "fr": "Comme le résultat est anormal, je vous conseille d'en parler à un cardiologue.",
    "en": "As the result is abnormal, I advise you to talk to a cardiologist.",
}
EMERGENCY_EN = ("If you have chest pain, fainting or strong shortness of breath, call the emergency number now "
                "(SAMU: 190 in Tunisia, 15 or 112 in France).")
EMERGENCY_WORDS = ["douleur thorac", "douleur a la poitrine", "douleur dans la poitrine", "mal a la poitrine",
                   "malaise", "evanou", "essouffl", "chest pain", "faint", "short of breath", "shortness of breath"]
DOCTOR_WORDS = ["medecin", "cardiologue", "doctor", "cardiologist", "urgence", "emergency"]

FRENCH_WORDS = ["le", "la", "les", "est", "que", "quoi", "comment", "mon", "ma", "mes", "je", "dois", "un", "une",
                "des", "pourquoi", "ce", "cette", "du", "de", "il", "elle", "et", "pas", "c'est", "qu'est"]
ENGLISH_WORDS = ["the", "is", "what", "how", "my", "do", "does", "should", "i", "you", "can", "why", "this", "it",
                 "and", "not", "are", "am", "to", "of", "give", "me", "please", "tell", "make", "an", "your", "with",
                 "for", "about", "explain", "recipe", "who", "where", "when"]


def simplify(text):
    # lower case, without accents, "œ" -> "oe": "Cœur" and "coeur", "Médecin" and "medecin" are the same word
    text = text.lower().replace("œ", "oe")
    text = unicodedata.normalize("NFD", text)
    return "".join([c for c in text if unicodedata.category(c) != "Mn"])


def has_word(text, words):
    # True if one of the words starts a word of the text ("cardi" finds "cardiaque" and "cardiologue")
    for word in words:
        if re.search(r"(^|[^a-z0-9])" + re.escape(word), text):
            return True
    return False


def language(question):
    # French or English: count the common small words of each language (French if equal)
    words = re.findall(r"[a-z']+", simplify(question))
    n_french = len([w for w in words if w in FRENCH_WORDS])
    n_english = len([w for w in words if w in ENGLISH_WORDS])
    if n_english > n_french:
        return "en"
    return "fr"


def is_on_topic(question, has_analysis=False):
    # -> (True / False, how it was decided). The keywords first (fast, always the same answer),
    # the LLM only for the questions that the lists cannot decide
    text = " " + simplify(question) + " "
    if has_word(text, HEART_WORDS):
        return True, "keyword"
    if has_word(text, OFF_TOPIC_WORDS):
        return False, "keyword"
    if has_analysis and (has_word(text, ANALYSIS_WORDS) or re.search(r"\d+[.,]\d+", text)):
        return True, "keyword"
    words = re.findall(r"[a-z']+", text)
    if 0 < len(words) <= 3 and has_word(text, GREETINGS):
        return True, "keyword"
    try:
        answer = llm.chat_json(prompts.CHAT_GUARD, "Question : " + question, max_tokens=20)
        return answer["content"].get("on_topic") is True, "llm"
    except Exception:
        return False, "llm unavailable"    # unsure and no LLM: the safe answer is the polite refusal


# ---------- 2. the analysis sent by the page ----------
def read_analysis(analysis):
    # the page sends the last answer of /analyze_pcg: {"audio": ..., "report": ...} or the audio result alone.
    # -> (audio result checked by Pydantic, report) or (None, None) if there is no valid analysis
    if not analysis:
        return None, None
    audio = analysis.get("audio") or analysis
    try:
        audio = AudioResult(**audio).model_dump()
    except Exception:
        return None, None
    report = analysis.get("report") if isinstance(analysis.get("report"), dict) else None
    return audio, report


def analysis_text(audio, report):
    # the facts of the analysis for the LLM (computed by the code, the LLM must not invent other numbers)
    lines = [facts_text(compute_facts(audio)), "- réponses des modèles pour ce son :"]
    for answer in audio["benchmark"]:
        if answer["probability_abnormal"] is None:
            probability = "pas de probabilité"
        else:
            probability = "probabilité d'anomalie " + str(answer["probability_abnormal"])
        lines.append("  * " + answer["model"] + " : " + LABELS[answer["prediction"]] + " (" + probability
                     + ", recall de validation " + str(answer["recall"]) + ")")
    if report and report.get("summary"):
        lines.append("- résumé du rapport des agents : " + report["summary"])
    return "\n".join(lines)


# ---------- 4. the safety checks of the answer ----------
def allowed_numbers(text):
    # the decimal numbers of the analysis, the question and the passages, also rounded (0.932 -> 0.93, 93.2 %, 93 %)
    numbers = set()
    for found in re.findall(r"\d+[.,]\d+", text):
        value = float(found.replace(",", "."))
        for shown in [value, round(value, 2), round(value, 1), round(value * 100, 1), round(value * 100)]:
            numbers.add(str(shown))
            numbers.add(("%.2f" % shown) if isinstance(shown, float) else str(shown))
    return numbers


def answer_problems(answer, allowed_text):
    # what the code refuses in an answer of the LLM
    problems = []
    lower = " " + answer.lower() + " "
    for word in FORBIDDEN:
        if word in lower:
            problems.append("diagnostic ou prescription : « " + word.strip() + " »")
    allowed = allowed_numbers(allowed_text)
    for found in re.findall(r"\d+[.,]\d+", answer):
        value = float(found.replace(",", "."))
        if str(value) not in allowed and ("%.2f" % value) not in allowed:
            problems.append("chiffre inventé : " + found)
    return problems


def safe_answer(lang, audio):
    # the answer written by the code when the LLM is down or its answer is refused
    if audio is None:
        if lang == "en":
            return "I can't answer this question precisely right now. For any question about your heart, talk to a doctor."
        return ("Je ne peux pas répondre précisément à cette question pour le moment. Pour toute question sur votre "
                "cœur, parlez-en à un médecin.")
    facts = compute_facts(audio)
    if lang == "en":
        label = {"normal": "normal", "abnormal": "abnormal"}[facts["prediction"]]
        return ("Your recording is classified as « " + label + " » by the main model (" + facts["model"]
                + "), with a probability of abnormality of " + "%.2f" % facts["probability_abnormal"]
                + ". This is not a diagnosis: only a doctor can confirm it.")
    return ("Votre enregistrement est classé « " + facts["label"] + " » par le modèle principal (" + facts["model"]
            + "), avec une probabilité d'anomalie de " + "%.2f" % facts["probability_abnormal"] + " (niveau de risque "
            + facts["risk_level"] + "). Ce n'est pas un diagnostic : seul un médecin peut le confirmer.")


def cited_sources(answer, sources_used, chunks):
    # the passages cited by the LLM (renumbered 1, 2, ... in the answer); none cited -> all the passages
    cited = []
    numbers = list(sources_used) if isinstance(sources_used, list) else []
    numbers = numbers + [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    for n in numbers:
        try:
            n = int(n)        # the LLM sometimes writes the numbers as texts: "1"
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(chunks) and n not in cited:
            cited.append(n)
    if not cited:
        sources = []
        for chunk in chunks:
            source = {"title": chunk["title"], "page": chunk["page"]}
            if source not in sources:
                sources.append(source)
        return answer, sources
    new_number = {}
    for i in range(len(cited)):
        new_number[cited[i]] = i + 1
    answer = re.sub(r"\[(\d+)\]", lambda m: "[" + str(new_number.get(int(m.group(1)), "?")) + "]", answer)
    answer = answer.replace("[?]", "")
    return answer, [{"title": chunks[n - 1]["title"], "page": chunks[n - 1]["page"]} for n in cited]


# ---------- the whole chat ----------
def chat(question, analysis=None, history=None):
    # -> (the ChatAnswer as a dict, the metrics for MLflow)
    start = time.time()
    lang = language(question)
    audio, report = read_analysis(analysis)
    metrics = {"prompt_tokens": 0, "eval_tokens": 0, "n_sources": 0, "safe_answer": False}

    on_topic, decided_by = is_on_topic(question, has_analysis=audio is not None)
    metrics["on_topic"] = on_topic
    metrics["guardrail"] = decided_by
    if not on_topic:
        metrics["total_s"] = round(time.time() - start, 2)
        return {"answer": REFUSAL[lang], "on_topic": False, "sources": [], "disclaimer": DISCLAIMER[lang]}, metrics

    text = " " + simplify(question) + " "
    if audio is None and has_word(text, MY_RECORDING):
        metrics["total_s"] = round(time.time() - start, 2)
        return {"answer": UPLOAD_FIRST[lang], "on_topic": True, "sources": [], "disclaimer": DISCLAIMER[lang]}, metrics

    # 3. RAG + the analysis -> the LLM
    chunks = safe_search(question, k=4, category="clinique")
    if audio is None:
        facts_part = "Aucun enregistrement n'a encore été analysé (question générale)."
    else:
        facts_part = analysis_text(audio, report)
    past = []
    for message in (history or [])[-6:]:
        who = "Utilisateur" if message["role"] == "user" else "Assistant"
        past.append(who + " : " + message["content"][:500])
    user_prompt = ("Langue de la réponse : " + ("anglais" if lang == "en" else "français") + "\n\n"
                   "Analyse de l'enregistrement (les seuls chiffres à utiliser) :\n" + facts_part + "\n\n"
                   "Conversation précédente :\n" + ("\n".join(past) if past else "(aucune)") + "\n\n"
                   "Extraits de documents :\n" + (chunks_text(chunks) if chunks else "(aucun extrait trouvé)") + "\n\n"
                   "Question : " + question)
    answer = None
    sources = []
    online_down = False
    try:
        reply = llm.chat_json(prompts.CHAT, user_prompt, max_tokens=MAX_TOKENS)
        metrics["prompt_tokens"] = reply["prompt_tokens"]
        metrics["eval_tokens"] = reply["eval_tokens"]
        text_answer = str(reply["content"].get("answer") or "").strip()
        allowed_text = facts_part + " " + question + " " + chunks_text(chunks)
        if text_answer != "" and not answer_problems(text_answer, allowed_text):
            answer, sources = cited_sources(text_answer, reply["content"].get("sources_used"), chunks)
    except llm.LLMUnavailable:
        online_down = True    # the online LLM cannot answer (key refused, no network): the page says so
    except Exception:
        pass                  # the local LLM is down or did not give JSON: the code answers from the analysis
    if answer is None:
        if online_down:
            answer = ONLINE_DOWN[lang]
        else:
            answer = safe_answer(lang, audio)
        metrics["safe_answer"] = True

    # 4. what the code always adds: a cardiologist for an abnormal result, the emergency number for alarm symptoms
    if audio is not None and audio["prediction"] == "abnormal" and not has_word(simplify(answer), DOCTOR_WORDS):
        answer = answer + " " + SEE_CARDIOLOGIST[lang]
    if has_word(text, EMERGENCY_WORDS) and "190" not in answer:
        answer = (EMERGENCY_EN if lang == "en" else EMERGENCY) + " " + answer

    metrics["n_sources"] = len(sources)
    metrics["total_s"] = round(time.time() - start, 2)
    return {"answer": answer, "on_topic": True, "sources": sources, "disclaimer": DISCLAIMER[lang]}, metrics
