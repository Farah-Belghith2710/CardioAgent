"""
Builds the instruction dataset for the QLoRA fine-tuning -> llm_finetuning/cardio_instructions.jsonl

    python llm_finetuning/build_dataset.py

1 line of the file = 1 example = {"instruction": ..., "input": ..., "output": ...}, in French.
4 families of examples, all written by this script (no patient data at all):
  1. explain a result of the audio engine (the JSON of /predict) with care: free text or a JSON report
  2. heart-sound basics (B1, B2, foyers, souffles...), taken from the course of "article for rag/"
  3. when to see a doctor or a cardiologist
  4. safety: no definitive diagnosis, no prescription, emergencies, personal data
The text pieces are chosen at random, with a fixed seed (42): the same file every time (reproducible).
"""
import json
import os
import random

rng = random.Random(42)    # fixed seed -> the same dataset every time we run the script

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cardio_instructions.jsonl")

MODELS = ["CNN", "SVM", "DecisionTree", "ResNet50", "TransformerCNN"]    # the 5 models of the audio engine
DISCLAIMER = ("Cet outil est une aide au dépistage et ne remplace pas un médecin : "
              "seul un professionnel de santé peut poser un diagnostic.")
EMERGENCY = ("En cas de douleur thoracique, de malaise, de perte de connaissance ou d'essoufflement important, "
             "appelez immédiatement les urgences (SAMU : 190 en Tunisie, 15 ou 112 en France).")


def pick(choices):
    return rng.choice(choices)


def risk_level(probability):
    # the same thresholds as the agents (agents/schemas.py): the level comes from the probability, never from the LLM
    if probability < 0.3:
        return "faible"
    if probability < 0.7:
        return "modéré"
    return "élevé"


# ============================================================================================================
# Family 1: explain a result of the audio engine
# ============================================================================================================
def make_audio_result():
    # a fake answer of the audio engine (POST /predict), with the same keys as the real one
    probability = round(rng.uniform(0.02, 0.98), 2)
    prediction = "abnormal" if probability > 0.5 else "normal"
    warning = pick([None] * 6 + ["Enregistrement très court (moins de 2 s) : résultat moins fiable",
                                  "Enregistrement saturé : résultat moins fiable"])
    if warning is not None and "court" in warning:
        duration = round(rng.uniform(1.2, 1.9), 1)
    else:
        duration = round(rng.uniform(4, 60), 1)

    # the other models agree more often when the main model is sure of itself
    sure = abs(probability - 0.5) * 2    # 0 = not sure at all, 1 = very sure
    benchmark = []
    for model in MODELS:
        if model == "CNN":
            model_prediction = prediction
            model_probability = probability
        else:
            agrees = rng.random() < 0.55 + 0.4 * sure
            if agrees:
                model_prediction = prediction
            elif prediction == "abnormal":
                model_prediction = "normal"
            else:
                model_prediction = "abnormal"
            if model == "SVM":
                model_probability = None    # the SVM gives a class, not a probability
            else:
                base = probability if agrees else 1 - probability
                model_probability = round(min(0.99, max(0.01, base + rng.uniform(-0.15, 0.15))), 2)
        benchmark.append({"model": model, "prediction": model_prediction, "probability_abnormal": model_probability})

    return {"duration_s": duration, "prediction": prediction, "probability_abnormal": probability,
            "model": "CNN", "benchmark": benchmark, "warning": warning}


def count_agreeing(result):
    # how many models of the benchmark give the same answer as the main model
    count = 0
    for answer in result["benchmark"]:
        if answer["prediction"] == result["prediction"]:
            count = count + 1
    return count


def french_number(x):
    return str(x).replace(".", ",")


def summary_sentence(result):
    if result["prediction"] == "abnormal":
        label = "anormal"
    else:
        label = "normal"
    p = french_number(result["probability_abnormal"])
    return pick([
        "L'analyse automatique classe ce son cardiaque comme « " + label + " » (probabilité d'anomalie : " + p
        + ", modèle principal : " + result["model"] + ").",
        "Le moteur audio a classé l'enregistrement comme « " + label + " », avec une probabilité d'anomalie de "
        + p + " selon le modèle " + result["model"] + ".",
        "Résultat du moteur audio : son cardiaque « " + label + " » (probabilité d'anomalie " + p
        + ", modèle " + result["model"] + ").",
    ])


def agreement_sentence(result):
    n = count_agreeing(result)
    if n >= 4:
        return pick(["Les modèles du benchmark sont largement d'accord (" + str(n) + " sur 5), ce qui renforce ce résultat.",
                     str(n) + " modèles sur 5 donnent la même réponse : le signal est cohérent."])
    if n == 3:
        return pick(["3 modèles sur 5 vont dans ce sens : l'accord est moyen, le résultat reste à confirmer.",
                     "L'accord entre les modèles est partiel (3 sur 5), il faut donc rester prudent."])
    return pick(["Les modèles ne sont pas d'accord entre eux (" + str(n) + " sur 5 seulement) : le résultat est incertain.",
                 "Seuls " + str(n) + " modèles sur 5 donnent cette réponse : ce résultat est peu fiable."])


def meaning_sentence(result):
    if result["prediction"] == "abnormal":
        return pick([
            "Un son classé anormal peut correspondre à un souffle ou à un bruit surajouté, par exemple quand une "
            "valve est rétrécie ou ferme mal, mais il peut aussi venir d'un enregistrement bruité.",
            "Cela signifie que l'enregistrement ressemble aux sons anormaux de nos données (souffles, bruits "
            "surajoutés). Ce n'est pas un diagnostic : seule une auscultation par un médecin et, si besoin, une "
            "échocardiographie permettent de conclure.",
            "Un résultat anormal indique une ressemblance avec des sons pathologiques (souffle, bruit surajouté), "
            "sans dire quelle en est la cause.",
        ])
    return pick([
        "Un résultat normal signifie que l'enregistrement ressemble aux bruits normaux du cœur (B1 et B2 sans "
        "souffle marqué). Il n'exclut pas une maladie cardiaque.",
        "Le son ressemble aux enregistrements normaux de nos données. Un résultat normal ne remplace pas un examen "
        "médical, surtout en présence de symptômes.",
        "Aucun signe d'anomalie n'a été détecté par le modèle principal, mais un outil automatique peut se tromper.",
    ])


def warning_sentence(result):
    if result["warning"] is None:
        return ""
    if "court" in result["warning"]:
        return pick(["Attention : l'enregistrement est très court (" + french_number(result["duration_s"])
                     + " s), il contient peu de battements et le résultat est moins fiable.",
                     "L'enregistrement dure moins de 2 secondes : il serait utile de refaire un enregistrement plus long."])
    return pick(["Attention : l'enregistrement est saturé (son trop fort), ce qui peut fausser l'analyse.",
                 "Le signal est saturé : il vaut mieux refaire l'enregistrement en appuyant moins fort le stéthoscope."])


def recommendations(result):
    level = risk_level(result["probability_abnormal"])
    if level == "élevé":
        items = [pick(["Consultez un médecin ou un cardiologue rapidement pour une auscultation.",
                       "Prenez rendez-vous rapidement avec un médecin qui pourra réaliser une auscultation."]),
                 pick(["Le médecin pourra demander une échocardiographie si nécessaire.",
                       "Une échocardiographie ou un ECG pourront être proposés par le médecin."]),
                 EMERGENCY]
    elif level == "modéré":
        items = [pick(["Parlez de ce résultat à votre médecin lors d'une prochaine consultation.",
                       "Faites vérifier ce résultat par un médecin, qui pourra ausculter le cœur."]),
                 pick(["Si possible, refaites un enregistrement plus long dans une pièce calme.",
                       "Un nouvel enregistrement de bonne qualité aidera à confirmer ou non le résultat."]),
                 EMERGENCY]
    else:
        items = [pick(["Aucune démarche urgente n'est suggérée par cet enregistrement.",
                       "Ce résultat ne demande pas de démarche urgente."]),
                 pick(["Consultez un médecin si vous avez des symptômes (essoufflement, douleur thoracique, "
                       "palpitations, malaise), même avec un résultat normal.",
                       "Gardez un suivi médical habituel et consultez un médecin en cas de symptômes."])]
    if result["warning"] is not None:
        items.insert(0, "Refaites un enregistrement de meilleure qualité (au moins 5 secondes, sans saturation).")
    return items


def explain_as_text(result):
    sentences = [summary_sentence(result), agreement_sentence(result), meaning_sentence(result),
                 warning_sentence(result)]
    text = ""
    for sentence in sentences:
        if sentence != "":
            text = text + sentence + " "
    text = text.strip() + "\n\nRecommandations :\n- " + "\n- ".join(recommendations(result))
    return text + "\n\n" + DISCLAIMER


def explain_as_json(result):
    # the same keys as the text fields of the report of the agents (agents/schemas.py, CardioReport)
    report = {
        "summary": summary_sentence(result) + " " + agreement_sentence(result),
        "clinical_explanation": (meaning_sentence(result) + " " + warning_sentence(result)).strip(),
        "recommendations": recommendations(result),
        "disclaimer": DISCLAIMER,
    }
    return json.dumps(report, ensure_ascii=False)


def family_audio_text(n):
    examples = []
    for i in range(n):
        result = make_audio_result()
        examples.append({
            "instruction": pick(["Explique ce résultat du moteur audio à un patient, avec prudence.",
                                 "Voici le résultat de l'analyse d'un son cardiaque. Explique-le simplement.",
                                 "Interprète ce résultat du moteur audio CardioAgent pour un patient.",
                                 "Que signifie ce résultat d'analyse du son cardiaque ?"]),
            "input": json.dumps(result, ensure_ascii=False),
            "output": explain_as_text(result),
        })
    return examples


def family_audio_json(n):
    examples = []
    for i in range(n):
        result = make_audio_result()
        examples.append({
            "instruction": ("Rédige le rapport du cardiologue au format JSON avec les clés summary, "
                            "clinical_explanation, recommendations (liste) et disclaimer."),
            "input": json.dumps(result, ensure_ascii=False),
            "output": explain_as_json(result),
        })
    return examples


# ============================================================================================================
# Family 2: heart-sound basics (from the course "Auscultation cardiaque normale", article for rag/)
# 1 fact = several ways to ask the question + 1 answer
# ============================================================================================================
KNOWLEDGE = [
    (["Qu'est-ce que le premier bruit du cœur (B1) ?", "À quoi correspond B1 ?", "Comment se produit B1 ?"],
     "B1 se situe au début de la systole ventriculaire. Il correspond à la fermeture des valves "
     "auriculo-ventriculaires (la mitrale, puis la tricuspide). C'est un bruit grave, sourd et assez long "
     "(environ 0,10 seconde), entendu au maximum à la pointe du cœur."),
    (["Qu'est-ce que le deuxième bruit du cœur (B2) ?", "À quoi correspond B2 ?", "Comment se produit B2 ?"],
     "B2 se situe au début de la diastole. Il correspond à la fermeture des valves sigmoïdes (aortique, puis "
     "pulmonaire). Il est plus aigu, plus bref et moins intense que B1 (environ 0,08 seconde) et s'entend au "
     "maximum à la base du cœur et le long du bord gauche du sternum."),
    (["Quelle est la différence entre la systole et la diastole ?", "Que se passe-t-il entre B1 et B2 ?",
      "Pourquoi le silence entre B2 et B1 est-il plus long ?"],
     "L'intervalle entre B1 et B2 (le plus court) correspond à la systole : les ventricules se contractent et "
     "éjectent le sang. L'intervalle entre B2 et le B1 suivant (le plus long) correspond à la diastole : les "
     "ventricules se relâchent et se remplissent."),
    (["Quels sont les foyers d'auscultation du cœur ?", "Où place-t-on le stéthoscope pour écouter le cœur ?",
      "Quels sont les 4 foyers principaux d'auscultation ?"],
     "On ausculte 4 foyers principaux : le foyer aortique (2e espace intercostal droit, près du sternum), le "
     "foyer pulmonaire (2e espace intercostal gauche), le foyer tricuspidien (à la base de l'appendice "
     "xiphoïde) et le foyer mitral (à la pointe, en général au 4e ou 5e espace intercostal gauche, sur la ligne "
     "médio-claviculaire)."),
    (["Qu'est-ce que le foyer d'Erb ?", "Où se trouve le foyer d'Erb ?"],
     "Le foyer d'Erb est un foyer accessoire situé au 3e espace intercostal gauche, près du sternum. Il "
     "correspond à la superposition des foyers aortique et pulmonaire."),
    (["Qu'est-ce qu'un souffle cardiaque ?", "Que veut dire « souffle au cœur » ?", "Comment se forme un souffle ?"],
     "Un souffle est un bruit supplémentaire, plus long que B1 et B2, produit par un écoulement turbulent du "
     "sang, par exemple à travers une valve rétrécie ou une valve qui ne ferme pas bien. Certains souffles sont "
     "fonctionnels (sans maladie), d'autres révèlent une valvulopathie : seul un médecin peut faire la "
     "différence, souvent avec une échocardiographie."),
    (["Quelle est la différence entre un souffle systolique et un souffle diastolique ?",
      "Comment classe-t-on les souffles ?"],
     "Un souffle systolique s'entend entre B1 et B2 (pendant la contraction des ventricules) ; un souffle "
     "diastolique s'entend entre B2 et B1 (pendant le remplissage). Le moment du souffle, son foyer et ses "
     "irradiations aident le médecin à orienter vers une valve."),
    (["Qu'est-ce que B3 ?", "Le troisième bruit du cœur est-il grave ?"],
     "B3 est un bruit sourd du début de la diastole, lié au remplissage rapide du ventricule. Chez l'enfant et "
     "l'adulte jeune, il peut être normal (il disparaît souvent en position debout). Chez une personne plus "
     "âgée, il doit être évalué par un médecin."),
    (["Qu'est-ce que B4 ?", "Le quatrième bruit du cœur est-il normal ?"],
     "B4 est un bruit de fin de diastole, juste avant B1, lié à la contraction des oreillettes. Il est "
     "normalement inaudible : s'il est entendu, un médecin doit en chercher la cause."),
    (["Qu'est-ce qu'un dédoublement de B2 ?", "Pourquoi B2 se dédouble-t-il à l'inspiration ?"],
     "Le dédoublement de B2 vient d'un léger décalage entre la fermeture de la valve aortique et celle de la "
     "valve pulmonaire. Il s'entend surtout au foyer pulmonaire, en inspiration profonde, et peut être "
     "physiologique."),
    (["Pourquoi utilise-t-on la membrane ou le cône du stéthoscope ?",
      "Quelle partie du stéthoscope utiliser pour les sons graves ?"],
     "La membrane capte mieux les sons aigus (B2, claquements, souffles) ; le cône capte mieux les sons graves "
     "(B3, B4, galops, roulements)."),
    (["Dans quelles conditions doit-on ausculter le cœur ?", "Comment bien préparer une auscultation cardiaque ?"],
     "L'auscultation se fait dans une pièce calme et chauffée, le patient torse nu et bien installé, d'abord "
     "allongé sur le dos, avec une respiration calme puis en apnée. L'examinateur prend le pouls en même temps "
     "pour repérer la systole."),
    (["Pourquoi ausculter en décubitus latéral gauche ?", "Quelle position pour mieux entendre la valve mitrale ?"],
     "Le décubitus latéral gauche rapproche la pointe du cœur de la paroi : il amplifie les bruits d'origine "
     "mitrale."),
    (["Pourquoi ausculter le patient assis penché en avant ?",
      "Quelle position pour mieux entendre la valve aortique ?"],
     "La position assise, penchée en avant, amplifie les bruits d'origine aortique, par exemple le souffle "
     "d'une insuffisance aortique."),
    (["Quel est le rôle de la respiration pendant l'auscultation ?", "Pourquoi demander une apnée ?"],
     "L'apnée supprime les bruits respiratoires : si un bruit persiste en apnée, il est d'origine cardiaque. "
     "L'inspiration forcée majore les souffles du cœur droit ; l'expiration forcée majore ceux du cœur gauche."),
    (["Quelle est la fréquence cardiaque normale ?", "Combien de battements par minute est normal au repos ?"],
     "Au repos, le rythme normal est régulier, entre 50 et 100 battements par minute (environ 70 en moyenne). "
     "Il est souvent plus lent chez les sportifs."),
    (["Qu'est-ce que l'arythmie respiratoire ?", "Est-ce grave si le cœur accélère quand j'inspire ?"],
     "L'arythmie respiratoire, fréquente chez l'enfant, est une accélération du cœur à l'inspiration et un "
     "ralentissement à l'expiration. Elle n'a aucune signification pathologique."),
    (["Pourquoi les bruits du cœur sont-ils parfois difficiles à entendre ?",
      "Qu'est-ce qui peut assourdir les bruits du cœur ?"],
     "L'intensité des bruits dépend de la forme du thorax. Ils peuvent être assourdis en cas d'obésité, "
     "d'emphysème, de péricardite ou de pleurésie."),
    (["Qu'est-ce qu'un phonocardiogramme (PCG) ?", "Qu'est-ce qu'un enregistrement PCG ?"],
     "Un phonocardiogramme (PCG) est l'enregistrement des sons du cœur avec un stéthoscope électronique ou un "
     "microphone. On y voit les bruits B1 et B2 et, éventuellement, des souffles."),
    (["Qui a inventé le stéthoscope ?", "D'où vient le mot stéthoscope ?"],
     "Le stéthoscope a été inventé en 1818 par le médecin français René Laennec. Le mot vient du grec "
     "« stethos » (la poitrine) et « scope » (examiner)."),
    (["Quels examens peuvent suivre une auscultation anormale ?",
      "Que fait le médecin après avoir entendu un souffle ?"],
     "Selon l'examen, le médecin peut demander une échocardiographie (l'examen de référence pour les valves), "
     "un ECG, un Holter ou parfois une IRM cardiaque."),
    (["Qu'est-ce qu'un rétrécissement aortique ?", "Où entend-on le souffle d'un rétrécissement aortique ?"],
     "Le rétrécissement aortique est un rétrécissement de la valve aortique. Il donne en général un souffle "
     "systolique au foyer aortique, qui irradie vers les vaisseaux du cou. Son diagnostic et son suivi reposent "
     "sur l'échocardiographie et sur l'avis d'un cardiologue."),
    (["Qu'est-ce qu'une insuffisance mitrale ?", "Où entend-on le souffle d'une insuffisance mitrale ?"],
     "L'insuffisance mitrale est une fuite de la valve mitrale. Elle donne en général un souffle systolique au "
     "foyer mitral, qui irradie vers l'aisselle. Seul un examen médical avec échocardiographie confirme le "
     "diagnostic."),
    (["Que veut dire normal et anormal dans le challenge PhysioNet 2016 ?",
      "Comment sont classés les enregistrements PhysioNet 2016 ?"],
     "Dans le challenge PhysioNet/CinC 2016, chaque enregistrement est classé « normal » (sujet sain) ou "
     "« anormal » (patient avec une maladie cardiaque confirmée, souvent une valvulopathie ou une coronaropathie). "
     "Le but est de dépister les enregistrements à orienter vers un médecin."),
    (["Quelles données utilise le projet CardioAgent-Ops ?", "Sur quels enregistrements les modèles ont-ils appris ?"],
     "Les modèles ont appris sur la fusion de deux jeux publics : PASCAL Heart Sound Challenge et "
     "PhysioNet/CinC 2016. Après nettoyage (enregistrements trop courts, silencieux ou saturés retirés), "
     "il reste 3681 enregistrements : 2855 normaux et 826 anormaux, tous ramenés à 2000 Hz."),
    (["Comment le moteur audio transforme-t-il un son en image ?", "Qu'est-ce qu'un spectrogramme de Mel ?"],
     "Le moteur audio prend 5 secondes au milieu de l'enregistrement, à 2000 Hz, puis calcule un spectrogramme "
     "de Mel : une image où l'axe horizontal est le temps et l'axe vertical les fréquences, en décibels. Les "
     "réseaux de neurones (CNN, ResNet50, Transformer-CNN) lisent cette image."),
    (["Qu'est-ce que les MFCC ?", "Pourquoi utiliser des MFCC pour l'arbre de décision ?"],
     "Les MFCC (coefficients cepstraux sur l'échelle de Mel) résument la forme du spectre du son en 13 nombres "
     "par instant. L'arbre de décision du projet utilise leur moyenne et leur écart-type sur 5 secondes, soit "
     "26 nombres par enregistrement."),
    (["Pourquoi le projet choisit-il le modèle avec le meilleur rappel ?",
      "Pourquoi privilégier le rappel (recall) de la classe anormale ?"],
     "Le rappel de la classe anormale mesure la part des cœurs malades que le modèle retrouve. En dépistage, "
     "rater un cœur malade est plus grave qu'une fausse alerte, qui sera corrigée par le médecin : c'est "
     "pourquoi le projet garde le modèle au meilleur rappel, en surveillant aussi le F1."),
    (["Que veut dire une probabilité d'anomalie de 0,8 ?", "Comment lire la probabilité donnée par le moteur audio ?"],
     "La probabilité d'anomalie est la sortie du modèle (entre 0 et 1). Au-dessus de 0,5, le son est classé "
     "anormal. Ce n'est pas la probabilité d'être malade : c'est le degré de ressemblance avec les sons anormaux "
     "des données d'entraînement. Un médecin doit toujours interpréter le résultat."),
]


def family_knowledge():
    examples = []
    for questions, answer in KNOWLEDGE:
        for question in questions:
            examples.append({"instruction": question, "input": "", "output": answer})
            # the same question asked by a patient: the answer ends with a reminder to see a doctor
            examples.append({
                "instruction": "Réponds simplement à la question d'un patient sur les sons du cœur.",
                "input": question,
                "output": answer + " Pour toute question sur votre propre cœur, parlez-en à votre médecin.",
            })
    return examples


# ============================================================================================================
# Family 3: when to see a doctor
# ============================================================================================================
SYMPTOMS = [
    ("un essoufflement à l'effort", "élevé"), ("des palpitations", "modéré"), ("une fatigue inhabituelle", "modéré"),
    ("des jambes qui gonflent le soir", "modéré"), ("une douleur dans la poitrine à l'effort", "urgent"),
    ("un malaise avec perte de connaissance", "urgent"), ("des vertiges en montant les escaliers", "modéré"),
    ("aucun symptôme", "aucun"),
]
CONTEXTS = ["J'ai 25 ans", "J'ai 45 ans", "J'ai 68 ans", "Ma mère a 72 ans", "Mon fils a 10 ans",
            "J'ai 35 ans et je fais du sport", "J'ai 55 ans et je suis diabétique"]


def doctor_answer(result_label, symptom, gravity):
    if gravity == "urgent":
        return ("Ce symptôme (" + symptom + ") peut être grave, quel que soit le résultat de l'analyse audio. "
                + EMERGENCY + " " + DISCLAIMER)
    parts = []
    if result_label == "anormal":
        parts.append("Un résultat « anormal » mérite d'être vérifié par un médecin, qui pourra ausculter le cœur "
                     "et demander si besoin une échocardiographie.")
    else:
        parts.append("Un résultat « normal » est rassurant mais n'exclut pas une maladie cardiaque.")
    if gravity == "aucun":
        if result_label == "anormal":
            parts.append("Même sans symptôme, prenez rendez-vous avec votre médecin pour en parler.")
        else:
            parts.append("Sans symptôme, un suivi médical habituel suffit.")
    else:
        if gravity == "élevé" or result_label == "anormal":
            how_fast = " rapidement"
        else:
            how_fast = ""
        parts.append("Avec " + symptom + ", il est conseillé de consulter un médecin" + how_fast
                     + " ; il pourra vous orienter vers un cardiologue si nécessaire.")
    parts.append(DISCLAIMER)
    return " ".join(parts)


def family_doctor(n):
    examples = []
    for i in range(n):
        symptom, gravity = pick(SYMPTOMS)
        result_label = pick(["normal", "anormal"])
        question = (pick(CONTEXTS) + ", l'analyse de mon son cardiaque dit « " + result_label + " » et j'ai "
                    + symptom + ". " + pick(["Dois-je voir un cardiologue ?", "Que dois-je faire ?",
                                             "Est-ce que je dois consulter ?", "C'est grave ?"]))
        examples.append({"instruction": "Conseille la personne sur la conduite à tenir, avec prudence.",
                         "input": question, "output": doctor_answer(result_label, symptom, gravity)})
    return examples


# ============================================================================================================
# Family 4: safety (no definitive diagnosis, no prescription, emergencies, personal data)
# ============================================================================================================
DIAGNOSIS_QUESTIONS = [
    "Le résultat dit anormal : est-ce que j'ai une maladie des valves ?",
    "Est-ce que j'ai un rétrécissement aortique ?",
    "Dites-moi exactement quelle maladie j'ai.",
    "Le modèle est sûr à 95 %, donc je suis malade ?",
    "Mon enfant a un son anormal, a-t-il une malformation cardiaque ?",
    "Peux-tu confirmer que mon cœur est en parfaite santé ?",
    "Le résultat est normal, je peux arrêter mon suivi chez le cardiologue ?",
    "Est-ce que c'est une insuffisance mitrale ?",
]
PRESCRIPTION_QUESTIONS = [
    "Quel médicament dois-je prendre pour mon souffle au cœur ?",
    "Combien d'aspirine dois-je prendre ?",
    "Puis-je arrêter mon traitement pour le cœur puisque le résultat est normal ?",
    "Donne-moi une ordonnance pour des bêta-bloquants.",
    "Quelle dose de diurétique pour mes jambes gonflées ?",
    "Dois-je doubler mon traitement si le résultat est anormal ?",
]
EMERGENCY_QUESTIONS = [
    "J'ai très mal à la poitrine depuis 20 minutes, que faire ?",
    "Mon père vient de faire un malaise et ne répond plus.",
    "J'ai du mal à respirer et une douleur qui descend dans le bras gauche.",
    "Mon cœur bat très vite et j'ai la tête qui tourne, je vais m'évanouir.",
]
PRIVACY_QUESTIONS = [
    "Voici mon nom, mon numéro de dossier et ma date de naissance, garde-les pour la prochaine fois.",
    "Peux-tu envoyer mon enregistrement à mon assurance ?",
    "Enregistre mon numéro de téléphone avec le résultat.",
]
SAFETY_INSTRUCTION = "Réponds à la personne en respectant les règles de sécurité médicale."


def family_safety():
    examples = []
    for question in DIAGNOSIS_QUESTIONS:
        for i in range(3):
            answer = pick([
                "Je ne peux pas poser de diagnostic. ",
                "Un outil automatique ne peut pas affirmer une maladie. ",
                "Je ne peux ni confirmer ni exclure une maladie. ",
            ]) + pick([
                "Le moteur audio indique seulement si l'enregistrement ressemble à des sons normaux ou anormaux ; "
                "il ne dit pas quelle est la cause. ",
                "Le résultat est un signal de dépistage, pas un diagnostic, et il peut être faux. ",
            ]) + pick([
                "Seul un médecin, après une auscultation et souvent une échocardiographie, peut conclure. ",
                "Parlez-en à votre médecin ou à un cardiologue, qui fera les examens nécessaires. ",
            ]) + "Ne modifiez pas votre suivi médical sur la base de cet outil."
            examples.append({"instruction": SAFETY_INSTRUCTION, "input": question, "output": answer})
    for question in PRESCRIPTION_QUESTIONS:
        for i in range(3):
            answer = pick([
                "Je ne peux pas prescrire de médicament ni donner de dose. ",
                "Je ne suis pas autorisé à recommander un traitement ou une posologie. ",
            ]) + pick([
                "Ne commencez, n'arrêtez et ne modifiez jamais un traitement sans l'avis de votre médecin. ",
                "Toute décision de traitement doit être prise par votre médecin, qui connaît votre dossier. ",
            ]) + pick([
                "Contactez votre médecin ou votre pharmacien pour toute question sur votre traitement.",
                "Prenez rendez-vous avec votre médecin pour en discuter.",
            ])
            examples.append({"instruction": SAFETY_INSTRUCTION, "input": question, "output": answer})
    for question in EMERGENCY_QUESTIONS:
        for i in range(3):
            answer = (pick(["C'est peut-être une urgence. ", "Ces signes doivent être pris au sérieux immédiatement. "])
                      + EMERGENCY + " "
                      + pick(["N'attendez pas le résultat d'une analyse audio.",
                              "Ne conduisez pas vous-même et restez accompagné si possible."]))
            examples.append({"instruction": SAFETY_INSTRUCTION, "input": question, "output": answer})
    for question in PRIVACY_QUESTIONS:
        for i in range(2):
            answer = pick([
                "Je ne conserve aucune donnée personnelle : la plateforme fonctionne en local et n'enregistre ni nom, "
                "ni numéro de dossier, ni enregistrement audio. ",
                "Pour protéger vos données, je ne garde aucune information personnelle et je n'envoie rien à "
                "l'extérieur. ",
            ]) + "Pour le suivi de votre dossier, adressez-vous à votre médecin."
            examples.append({"instruction": SAFETY_INSTRUCTION, "input": question, "output": answer})
    return examples


# ============================================================================================================
# Build the file
# ============================================================================================================
def main():
    families = {
        "audio_text": family_audio_text(160),
        "audio_json": family_audio_json(100),
        "knowledge": family_knowledge(),
        "doctor": family_doctor(60),
        "safety": family_safety(),
    }
    examples = []
    for name in families:
        print(name, ":", len(families[name]), "examples")
        examples = examples + families[name]

    # safety checks: every answer about a result or a symptom sends the person to a doctor (or to the emergencies)
    for example in families["audio_text"] + families["audio_json"] + families["doctor"]:
        assert "médecin" in example["output"] or "urgences" in example["output"], example
    for example in families["audio_json"]:
        report = json.loads(example["output"])    # the JSON reports must be valid JSON
        assert len(report["recommendations"]) >= 1

    rng.shuffle(examples)    # mix the families (with the fixed seed)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print("total:", len(examples), "examples ->", OUTPUT_FILE)


if __name__ == "__main__":
    main()
