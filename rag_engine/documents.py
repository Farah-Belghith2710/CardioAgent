"""
The documents of the RAG (folder "article for rag/"): a readable title and a category for each PDF.
  - "clinique"  : medical courses and recommendations -> used by the Cardiologue agent
  - "technique" : papers about heart-sound classification algorithms -> may be used by the Analyste agent
A new PDF dropped in the folder (not in this list) gets its file name as title and the category "clinique".
The full references (authors, year, URL, license) are in "article for rag/SOURCES.md".
"""

DOCUMENTS = {
    # clinique
    "Auscultation-cardiaque-normale-4-4.pdf":
        ("Auscultation cardiaque normale - Dr F. Touati, Univ. Constantine 3 (2024-2025)", "clinique"),
    "Auscultation-pathologique-bruits-surajoutes-Chettibi-2023-2024.pdf":
        ("Les bruits surajoutés en auscultation cardiaque - Dr Y. Chettibi, Univ. Constantine 3 (2023-2024)", "clinique"),
    "UMVF-semiologie-cardiologique-examen-clinique.pdf":
        ("Sémiologie cardiologique : examen clinique - UMVF (2008-2009)", "clinique"),
    "Hal open science.pdf":
        ("Huette & Abou-Arab (2023) - Les nouvelles recommandations de l'ESC sur l'évaluation cardiovasculaire",
         "clinique"),
    "ehaf194.pdf":
        ("2025 ESC/EACTS Guidelines for the management of valvular heart disease", "clinique"),
    # technique: the PhysioNet/CinC Challenge 2016 and 12 of its entries
    "challenge2016.pdf":
        ("Clifford et al. (2016) - Classification of Normal/Abnormal Heart Sound Recordings: "
         "the PhysioNet/CinC Challenge 2016", "technique"),
    "goda.pdf": ("Goda & Hajas (2016) - Morphological Determination of Pathological PCG Signals", "technique"),
    "grzegorczyk.pdf": ("Grzegorczyk et al. (2016) - PCG Classification Using a Neural Network Approach", "technique"),
    "homsi.pdf": ("Homsi et al. (2016) - Automatic Heart Sound Recording Classification using a Nested Set of "
                  "Ensemble Algorithms", "technique"),
    "langley.pdf": ("Langley & Murray (2016) - Abnormal Heart Sounds Detected from Short Duration Unsegmented "
                    "Phonocardiograms by Wavelet Entropy", "technique"),
    "nilanon.pdf": ("Nilanon et al. (2016) - Normal/Abnormal Heart Sound Recordings Classification Using "
                    "Convolutional Neural Network", "technique"),
    "ortiz.pdf": ("Gonzalez Ortiz et al. (2016) - Heart Sound Classification Based on Temporal Alignment Techniques",
                  "technique"),
    "potes.pdf": ("Potes et al. (2016) - Ensemble of Feature-based and Deep learning-based Classifiers for Detection "
                  "of Abnormal Heart Sounds", "technique"),
    "rubin.pdf": ("Rubin et al. (2016) - Classifying Heart Sound Recordings using Deep Convolutional Neural Networks "
                  "and Mel-Frequency Cepstral Coefficients", "technique"),
    "singhmiller.pdf": ("Singh-Miller & Singh-Miller (2016) - Using Spectral Acoustic Features to Identify Abnormal "
                        "Heart Sounds", "technique"),
    "tschannen.pdf": ("Tschannen et al. (2016) - Heart Sound Classification Using Deep Structured Features",
                      "technique"),
    "vernekar.pdf": ("Vernekar et al. (2016) - A Novel Approach for Classification of Normal/Abnormal "
                     "Phonocardiogram Recordings using Temporal Signal Analysis and Machine Learning", "technique"),
    "zabihi.pdf": ("Zabihi et al. (2016) - Heart Sound Anomaly and Quality Detection using Ensemble of Neural "
                   "Networks without Segmentation", "technique"),
}


def describe(file_name):
    # (title, category) of a PDF of the folder
    if file_name in DOCUMENTS:
        return DOCUMENTS[file_name]
    return (file_name.replace(".pdf", ""), "clinique")
