"""
RAG - step 3 (evaluation): does the search find the right document for questions whose answer we know?

    python -m rag_engine.evaluate_rag            (repo root, after rag_engine.ingest, Qdrant running)

18 questions in French (12 "clinique", 6 "technique"), each with the documents that contain the answer and a few
words that the answer should contain. For each question we take the K = 5 closest chunks and measure:
  - hit@k : is a chunk of an expected document among the k first chunks? (k = 1, 3, 5)
  - MRR   : mean of 1 / (rank of the first chunk of an expected document)  (0 if not in the 5 first)
  - keyword hit : does one of the 5 chunks contain one of the expected words?
The results are saved in rag_engine/rag_evaluation.json (for the report).
"""
import json
import os

from rag_engine.search import search

K = 5
NORMAL = "Auscultation-cardiaque-normale-4-4.pdf"
CHETTIBI = "Auscultation-pathologique-bruits-surajoutes-Chettibi-2023-2024.pdf"
UMVF = "UMVF-semiologie-cardiologique-examen-clinique.pdf"
HAL = "Hal open science.pdf"

# (category, question, expected documents, expected words)
QUESTIONS = [
    ("clinique", "À quoi correspond le premier bruit du cœur B1 ?", [NORMAL, UMVF], ["fermeture", "mitral"]),
    ("clinique", "Quels sont les quatre foyers d'auscultation cardiaque ?", [NORMAL, UMVF], ["foyer", "aortique"]),
    ("clinique", "Où se trouve le foyer d'Erb ?", [NORMAL], ["erb"]),
    ("clinique", "Pourquoi ausculter le cœur en décubitus latéral gauche ?", [NORMAL, UMVF], ["latéral gauche"]),
    ("clinique", "Qu'est-ce que le troisième bruit du cœur B3 et quand est-il normal ?", [NORMAL, UMVF, CHETTIBI],
     ["b3"]),
    ("clinique", "Qu'est-ce qu'un bruit de galop à l'auscultation cardiaque ?", [CHETTIBI, UMVF], ["galop"]),
    ("clinique", "Qu'est-ce que le claquement d'ouverture de la valve mitrale ?", [CHETTIBI, UMVF], ["claquement"]),
    ("clinique", "Comment reconnaître un frottement péricardique à l'auscultation ?", [CHETTIBI, UMVF], ["frottement"]),
    ("clinique", "Quelle est la différence entre la membrane et le cône du stéthoscope ?", [NORMAL, UMVF],
     ["membrane", "cône"]),
    ("clinique", "Qu'est-ce qu'un dédoublement du deuxième bruit du cœur B2 ?", [NORMAL, UMVF, CHETTIBI],
     ["dédoubl"]),
    ("clinique", "Quelles sont les recommandations de l'ESC pour l'évaluation cardiovasculaire avant une chirurgie "
                 "non cardiaque ?", [HAL], ["chirurgie", "esc"]),
    ("clinique", "Quels examens complémentaires demander après une auscultation cardiaque anormale ?",
     [NORMAL, UMVF, HAL], ["échocardiographie", "ecg"]),
    ("technique", "Combien d'enregistrements contient la base de données du challenge PhysioNet/CinC 2016 ?",
     ["challenge2016.pdf"], ["recordings"]),
    ("technique", "Comment le score du challenge PhysioNet 2016 est-il calculé avec la sensibilité et la "
                  "spécificité ?", ["challenge2016.pdf"], ["sensitivity", "specificity"]),
    ("technique", "Réseau de neurones convolutif appliqué aux coefficients MFCC des sons cardiaques",
     ["rubin.pdf", "nilanon.pdf", "potes.pdf"], ["convolutional", "mfcc"]),
    ("technique", "Classification des sons cardiaques par entropie en ondelettes sans segmentation",
     ["langley.pdf"], ["wavelet", "entropy"]),
    ("technique", "Quelles caractéristiques temps-fréquence ont été extraites pour détecter les sons cardiaques "
                  "anormaux ?", ["potes.pdf", "homsi.pdf", "goda.pdf", "vernekar.pdf", "singhmiller.pdf"],
     ["features"]),
    ("technique", "Détection de la qualité du signal PCG avec un ensemble de réseaux de neurones",
     ["zabihi.pdf"], ["quality"]),
]


def evaluate():
    rows = []
    for category, question, expected, words in QUESTIONS:
        results = search(question, k=K, category=category)
        rank = 0    # the rank of the first chunk of an expected document (0 = not found in the K first)
        for i in range(len(results)):
            if results[i]["source"] in expected:
                rank = i + 1
                break
        keyword_found = False
        for result in results:
            for word in words:
                if word in result["text"].lower():
                    keyword_found = True
        rows.append({
            "category": category,
            "question": question,
            "expected": expected,
            "retrieved": [r["source"] + " p." + str(r["page"]) for r in results],
            "rank": rank,
            "keyword_found": keyword_found,
        })
        print("[rank {}] {}".format(rank if rank > 0 else "-", question))
    return rows


def summarize(rows):
    summary = {"n_questions": len(rows), "k": K}
    for k in [1, 3, 5]:
        hits = 0
        for row in rows:
            if 0 < row["rank"] <= k:
                hits = hits + 1
        summary["hit@" + str(k)] = round(hits / len(rows), 3)
    reciprocal_ranks = []
    keyword_hits = 0
    for row in rows:
        if row["rank"] > 0:
            reciprocal_ranks.append(1 / row["rank"])
        else:
            reciprocal_ranks.append(0)
        if row["keyword_found"]:
            keyword_hits = keyword_hits + 1
    summary["mrr"] = round(sum(reciprocal_ranks) / len(rows), 3)
    summary["keyword_hit"] = round(keyword_hits / len(rows), 3)
    return summary


if __name__ == "__main__":
    rows = evaluate()
    summary = {"all": summarize(rows)}
    for category in ["clinique", "technique"]:
        part = [row for row in rows if row["category"] == category]
        summary[category] = summarize(part)
    print(json.dumps(summary, indent=2))
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_evaluation.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "questions": rows}, f, indent=2, ensure_ascii=False)
    print("Saved:", output_file)
