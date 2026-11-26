# Sources of the RAG (folder `article for rag/`)

Every PDF of this folder is indexed by `python -m rag_engine.ingest` (chunks of ~800 characters, 150 in common,
embeddings `paraphrase-multilingual-MiniLM-L12-v2`, Qdrant collection `cardio_guidelines`). Drop a new PDF here and
run the ingest again. The titles and categories used by the code are in `rag_engine/documents.py`.

**Copyright:** only the two PDFs already in git stay in git. Every other PDF is kept locally only
(`.gitignore`: `article for rag/*.pdf`), for personal and educational use.

| # | File | Title | Authors / origin | Year | URL | License / use | Category |
|---|---|---|---|---|---|---|---|
| 1 | `Auscultation-cardiaque-normale-4-4.pdf` | Auscultation cardiaque normale (UEI sémiologie cardio-vasculaire, 3e année) | Dr F. Touati, Faculté de médecine, Univ. Salah Boubnider Constantine 3 | 2024-2025 | course PDF of the faculty (already in this repo) | educational course material | clinique |
| 2 | `Hal open science.pdf` | Les nouvelles recommandations de la société européenne de cardiologie sur l'évaluation cardiovasculaire changent-elles ma pratique ? (Praticien en Anesthésie Réanimation) | B. Huette, O. Abou-Arab | 2023 | https://u-picardie.hal.science/hal-04166622 | CC BY-NC 4.0 | clinique |
| 3 | `Auscultation-pathologique-bruits-surajoutes-Chettibi-2023-2024.pdf` | Auscultation pathologique : les bruits surajoutés en auscultation cardiaque | Dr Y. Chettibi, Faculté de médecine, Univ. Constantine 3 | 2023-2024 | https://facmed.univ-constantine3.dz/wp-content/uploads/2023/11/Auscultation-pathologique-bruits-surajout%C3%A9s-Dr-Y.-Chettibi-2023-2024.pdf | educational course material (local copy only) | clinique |
| 4 | `UMVF-semiologie-cardiologique-examen-clinique.pdf` | Sémiologie cardiologique : examen clinique (support de cours) | Université Médicale Virtuelle Francophone (UMVF), archive UNESS | 2008-2009 | https://archives.uness.fr/sites/campus-unf3s-2014/semiologie-cardiologique/enseignement/examenclinique/site/html/cours.pdf | © UMVF, educational use (local copy only) | clinique |
| 5 | `challenge2016.pdf` | Classification of Normal/Abnormal Heart Sound Recordings: the PhysioNet/Computing in Cardiology Challenge 2016 | G. D. Clifford, C. Liu, B. Moody, D. Springer, I. Silva, Q. Li, R. G. Mark | 2016 | https://physionet.org/content/challenge-2016/ | Computing in Cardiology 2016 (open access) | technique |
| 6 | `goda.pdf` | Morphological Determination of Pathological PCG Signals by Time and Frequency Domain Analysis | M. Á. Goda, P. Hajas | 2016 | papers of the challenge (PhysioNet) | Computing in Cardiology 2016 | technique |
| 7 | `grzegorczyk.pdf` | PCG Classification Using a Neural Network Approach | I. Grzegorczyk et al. | 2016 | papers of the challenge (PhysioNet) | Computing in Cardiology 2016 | technique |
| 8 | `homsi.pdf` | Automatic Heart Sound Recording Classification using a Nested Set of Ensemble Algorithms | M. N. Homsi et al. | 2016 | papers of the challenge (PhysioNet) | Computing in Cardiology 2016 | technique |
| 9 | `langley.pdf` | Abnormal Heart Sounds Detected from Short Duration Unsegmented Phonocardiograms by Wavelet Entropy | P. Langley, A. Murray | 2016 | papers of the challenge (PhysioNet) | Computing in Cardiology 2016 | technique |
| 10 | `nilanon.pdf` | Normal / Abnormal Heart Sound Recordings Classification Using Convolutional Neural Network | T. Nilanon, J. Yao, J. Hao, S. Purushotham, Y. Liu | 2016 | papers of the challenge (PhysioNet) | Computing in Cardiology 2016 | technique |
| 11 | `ortiz.pdf` | Heart Sound Classification Based on Temporal Alignment Techniques | J. J. González Ortiz, C. P. Phoo, J. Wiens | 2016 | papers of the challenge (PhysioNet) | Computing in Cardiology 2016 | technique |
| 12 | `potes.pdf` | Ensemble of Feature-based and Deep learning-based Classifiers for Detection of Abnormal Heart Sounds | C. Potes, S. Parvaneh, A. Rahman, B. Conroy | 2016 | papers of the challenge (PhysioNet) | Computing in Cardiology 2016 | technique |
| 13 | `rubin.pdf` | Classifying Heart Sound Recordings using Deep Convolutional Neural Networks and Mel-Frequency Cepstral Coefficients | J. Rubin et al. | 2016 | papers of the challenge (PhysioNet) | Computing in Cardiology 2016 | technique |
| 14 | `singhmiller.pdf` | Using Spectral Acoustic Features to Identify Abnormal Heart Sounds | N. E. Singh-Miller, N. Singh-Miller | 2016 | papers of the challenge (PhysioNet) | Computing in Cardiology 2016 | technique |
| 15 | `tschannen.pdf` | Heart Sound Classification Using Deep Structured Features | M. Tschannen et al. | 2016 | papers of the challenge (PhysioNet) | Computing in Cardiology 2016 | technique |
| 16 | `vernekar.pdf` | A Novel Approach for Classification of Normal/Abnormal Phonocardiogram Recordings using Temporal Signal Analysis and Machine Learning | S. Vernekar, S. Nair, D. Vijaysenan, R. Ranjan | 2016 | papers of the challenge (PhysioNet) | Computing in Cardiology 2016 | technique |
| 17 | `zabihi.pdf` | Heart Sound Anomaly and Quality Detection using Ensemble of Neural Networks without Segmentation | M. Zabihi, A. B. Rad, S. Kiranyaz, M. Gabbouj, A. K. Katsaggelos | 2016 | papers of the challenge (PhysioNet) | Computing in Cardiology 2016 | technique |

## Not included: the 2025 ESC/EACTS Guidelines for the management of valvular heart disease

Praz F., Borger M. A. et al., European Heart Journal 2025, doi:10.1093/eurheartj/ehaf194
(https://www.escardio.org/Guidelines/Clinical-Practice-Guidelines/valvular-heart-disease).
The publisher's site blocks scripted downloads, and the ESC states that ESC Guidelines are protected by copyright
and that their use in software tools, generative AI models or algorithms requires a licence agreement with the ESC.
So this PDF is **not** in the folder. If you decide to add it for personal, educational use, save it here as
`ehaf194.pdf` (the code already knows its title and category, `clinique`) and run the ingest again.
The ESC recommendations are present in the index through document 2 (the French summary of the ESC 2022
recommendations, CC BY-NC 4.0).
