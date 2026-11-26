# MERGE_NOTES — merging Zaineb's work into this repo

Farah Belghith and Zaineb (`zaineb1802`) both worked on CardioAgent-Ops: Farah in this repo, Zaineb in a private repo
(`zaineb1802/private`, 2 commits, 6 notebooks + an API). This file explains what was compared, what was kept from whom,
and why. Only code was taken from Zaineb's repo (adapted to this repo's conventions): no data, no caches, no model
weights, and none of her git history.

## 1. How the two works were compared (fairly)

The two works used different protocols, so their scores could not be compared directly:

| | Farah | Zaineb |
|---|---|---|
| Data | 3766 wavs → **3681 clean wavs** (too short / silent / saturated removed) | 3784 wavs, no cleaning (193 abnormal PASCAL files vs 175) |
| Signal | 2000 Hz, the middle **5 s** (shorter wavs repeated) | 2000 Hz, the whole wav (SVM/tree) or the middle 5 s (CNN/ResNet) |
| Folds | `KFold(n_splits=3, shuffle=True, random_state=1)` | `StratifiedGroupKFold(3)` by patient, seed 42 |
| Keras backend | TensorFlow | PyTorch |

So every method of Zaineb was **run again on Farah's protocol**: the same 3681 clean wavs, in the same order, the same
3 folds, the same metrics (abnormal = positive class), TensorFlow backend (Keras 3 code is the same). All of it is in
[`notebooks/merge_farah_vs_zaineb.ipynb`](notebooks/merge_farah_vs_zaineb.ipynb) (executed, with outputs), figure:
`models plots/merge_farah_vs_zaineb.png`.

**Decision rule, in this order:** (a) matches the PDF, (b) runs end to end on this PC, (c) the highest mean validation
**recall** of abnormal (then F1), (d) simple enough to explain at the oral.

### Results (mean of the 3 folds, same data, same folds)

| Model | Method | Recall | F1 | Precision | Accuracy |
|---|---|---|---|---|---|
| SVM | Farah: Mel 30 bands flattened (5 s), rbf | 0.829 | 0.639 | 0.521 | 0.791 |
| SVM | **Zaineb: 72 features, rbf** | **0.887** | **0.702** | 0.581 | 0.831 |
| SVM | Zaineb: 22 basic features, rbf | 0.883 | 0.653 | 0.518 | 0.789 |
| SVM | Zaineb: 72 features, rbf, C/gamma tuned (GridSearchCV) | 0.795 | 0.737 | 0.687 | 0.872 |
| Decision tree | Farah: 26 MFCC, gini, depth 10 (chosen on a 33 % split) | 0.718 | 0.629 | 0.559 | 0.810 |
| Decision tree | Farah's 26 MFCC + Zaineb's depth choice (CV on the train part) | 0.825 | 0.630 | 0.515 | 0.781 |
| Decision tree | **Zaineb: 72 features, depth chosen by CV on the train part** | **0.863** | **0.656** | 0.531 | 0.797 |
| CNN | **Farah: Conv 16/32/64 + Flatten + Dense 32, 20 epochs** | **0.925** | 0.655 | 0.510 | 0.778 |
| CNN | Zaineb: BatchNorm + GlobalAveragePooling + softmax, early stopping | 0.798 | 0.705 | 0.632 | 0.850 |
| ResNet50 | **Farah: magma image 224 × 224, head BN + Dense 128** | **0.800** | **0.763** | 0.730 | 0.889 |
| ResNet50 | Zaineb: grey image 128 × 128, head Dense 256, early stopping | 0.757 | 0.740 | 0.723 | 0.881 |

Zaineb's 72 features (on the whole wav): duration, SNR, mean and std of 20 Mel bands, mean and std of 13 MFCC,
mean and std of the spectral centroid, mean and std of the zero-crossing rate.

## 2. Inventory against the PDF deliverables

| What the PDF asks | Farah has | Zaineb has | Quality notes | Decision |
|---|---|---|---|---|
| `01_eda_data_wrangling.ipynb`: fusion, cleaning, EDA | `01_Tp1_and_Tp2`: fusion, durations, SNR, FFT, Mel, KDE, ZCR | `01_Tp1_and_Tp2`: same starting notebook, no ZCR, no cleaning | Farah's is more complete | **Farah's** (+ cleaning, the fixed test set and Zaineb's patient-overlap check added) |
| Model 1: Decision tree on 1D features (MFCC) | 26 MFCC mean + std (5 s), gini, depth 10 chosen on a test split | 72 1D features (incl. the 26 MFCC), depth chosen by CV on the train part only | Zaineb wins on recall and F1; her depth choice never looks at the test wavs | **Merged**: Zaineb's depth choice by CV on the train part; her 72 features are compared with the 26 MFCC in notebook 03 and the best one on validation is kept |
| Model 2: SVM with an RBF kernel | Mel spectrogram flattened, duration study 2 / 5 / 10 s | 72 features, GridSearchCV, patient split | Zaineb's features win on recall and F1 (default C/gamma: tuning lowers the recall) | **Merged**: Farah's duration study is kept (data mining); Zaineb's 72 features are compared in notebook 02 and the best one on validation is kept |
| Model 3: custom CNN on Mel spectrograms | Conv 16/32/64, 20 epochs | BatchNorm + GAP + softmax, early stopping, PyTorch backend | Farah has the higher recall (0.925 vs 0.798), Zaineb the higher F1 | **Farah's** (recall first, as the PDF asks) |
| Model 4: pre-trained CNN (ResNet / VGG) | ResNet50 + VGG16, frozen, magma 224 × 224 | ResNet50, frozen, grey 128 × 128 | Farah wins on recall and F1 | **Farah's**; Zaineb's idea of doing the image preparation **inside the model** is reused for the ONNX export (notebook 05) |
| Model 5: Transformer-CNN hybrid | — | — | missing in both | built in `notebooks/07_transformer_cnn.ipynb` |
| MLflow tracking | inline logging in each notebook, 1 nested run per fold | `mlflow_helper.py`: 1 registered model per notebook, mean + std | Farah's shows the 3 folds (asked by the prof) and is simpler to explain | **Farah's** |
| ONNX export + `cardio_model.onnx` | notebook 05 (skl2onnx, tf2onnx) | none (models served from the MLflow registry with PyTorch) | the PDF asks for ONNX | **Farah's** |
| Service 3 `audio-engine` (ONNX Runtime + Librosa) | FastAPI + ONNX Runtime, 3 models | none (prediction done in her gateway) | the PDF puts inference in the audio engine | **Farah's** (+ Zaineb's 72-feature function, for the SVM and the tree) |
| Service 5 `api-gateway` (`/analyze_pcg`) | FastAPI router + her web page | FastAPI that loads the models from MLflow (PyTorch inside the gateway), best model by F1 | the PDF: the gateway forwards; the official model = best recall | **Farah's** |
| Service 6 `mlflow-server` | Dockerfile, MLflow 2.22.4 (same as the client) | compose service, MLflow 2.16.2 | client and server versions should match | **Farah's** |
| `docker-compose.yml` (6 services) | none | 2 services (mlflow + api) | | new file (Phase 8) |
| `tests/` (pytest) | none | `test_cnn_inference.py` (unittest, wav generated with soundfile, mocked MLflow) | good ideas | ideas reused in `tests/` (Phase 9) |
| Smoke test with real wavs | none | `check_api.py` | useful for the end-to-end check | idea reused for the end-to-end check of the platform (Phase 8) |
| 5 pre-trained backbones compared | ResNet50 + VGG16 (notebook 06) | `05_Tp7_transfer_learning_compare` (never executed: no results) | | not merged |
| `llm-server`, `vector-db`, `agent-orchestrator`, `llm_finetuning/`, `rag_engine/`, `agents/`, CI | none | none | | built in Phases 5–9 |

## 3. Decisions, one by one

1. **EDA notebook** — Farah's notebook is kept (it already had the ZCR study). The cleaning rules and the fixed
   20 % test set were added to it, plus a small check taken from Zaineb's idea of patient leakage (PASCAL wavs of the
   same patient in train and test).
2. **SVM** — Zaineb's 72 features beat the flattened Mel spectrogram on the same folds (recall 0.887 vs 0.829, F1
   0.702 vs 0.639). Her feature function now lives in notebook 02 next to Farah's features; the notebook keeps the
   feature set with the best validation recall. Her tuned version was not kept: it trades recall for precision.
3. **Decision tree** — Zaineb's method (depth chosen by a cross-validation on the training part only) is cleaner than
   choosing the depth on a split that contains the test wavs, and it alone raised the recall of the MFCC tree from
   0.718 to 0.825. With her 72 features the recall reaches 0.863. Notebook 03 uses her depth choice and compares the
   26 MFCC (what the PDF names) with the 72 features (which contain the same 26 MFCC).
4. **CNN** — Farah's CNN stays: it finds more sick hearts (recall 0.925 vs 0.798), which is the project's rule.
5. **ResNet50** — Farah's version stays (better recall and F1). Zaineb's trick of putting the image preparation
   (resize, ImageNet colours) inside the Keras model is reused so the ONNX file needs no TensorFlow in the engine.
6. **Folds** — the prof's `KFold(3, shuffle=True, random_state=1)` is kept for every model, so all models are
   compared on the same folds. Zaineb's patient-grouped folds are a good idea, but only the PASCAL file names give
   the patient (PhysioNet names do not), and her own experiment showed a difference of only 0.005 in F1.
7. **Services** — Farah's architecture follows the PDF (ONNX inference in the audio engine, a thin gateway); Zaineb's
   gateway needed PyTorch and MLflow inside it, so it was not merged.

## 4. What came from Zaineb (credits)

| Zaineb's piece | Where it is now |
|---|---|
| The 72-feature extraction (`get_wav_features`) | `notebooks/02_Tp3_svm.ipynb`, `notebooks/03_Tp4_decision_tree.ipynb`, audio engine `features_extended()` |
| Depth chosen by cross-validation on the training part only | `notebooks/03_Tp4_decision_tree.ipynb` |
| Image preparation inside the transfer-learning model | ONNX export of ResNet50 in `notebooks/05_best_model_onnx.ipynb` |
| Patient-leakage awareness | the patient-overlap check in `notebooks/01_eda_data_wrangling.ipynb` |
| Test style (wav generated with soundfile, mocked services) | `tests/` |
| Smoke test of the API with real wavs | end-to-end check of the platform (Phase 8) |
