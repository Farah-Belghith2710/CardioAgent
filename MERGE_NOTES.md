# MERGE_NOTES — merging Zaineb's work into this repo

Farah Belghith and Zaineb (`zaineb1802`) both worked on CardioAgent-Ops: Farah in this repo, Zaineb in a private repo
(`zaineb1802/private`, 2 commits, 6 notebooks + an API). This file explains what was compared, what was merged, what
won, and why. Only code and ideas were taken from Zaineb's repo, adapted to this repo's conventions: no data, no caches,
no model weights, none of her git history.

**What "merge" means here:** for each model, the final notebook in `notebooks/` *is* the best version — Farah's,
Zaineb's, or a mix of the two that does better than both. Zaineb's notebooks could not be copied as they were (other
data: 3784 wavs without cleaning; other folds: grouped by patient; PyTorch backend), so the winning method was put
inside the common notebook, with the common protocol. The detailed proof is
[`notebooks/09_merge_farah_vs_zaineb.ipynb`](notebooks/09_merge_farah_vs_zaineb.ipynb) (executed, with outputs):
part 1 = the first comparison, part 2 = the real merge. Its results are saved in `models plots/merge_results.json`,
which the notebooks 02, 03, 04 and 06 show in their first cell ("Merge: Farah vs Zaineb") before training only the
winner.

## 1. The rule to decide "better"

- Same data (the 3681 clean wavs), same 80 % train part, **same 3 validation folds**
  (`KFold(3, shuffle=True, random_state=1)`), same metrics (abnormal = the positive class). **Never** the 20 % test set.
- A candidate **replaces** the current version if it has a higher validation **recall**, or a higher **F1** without
  losing more than 0.01 of recall. When a candidate has more recall but loses more than 0.01 of F1 (or the opposite),
  it is a **trade-off**: we keep the higher recall (the rule of the PDF) and write it below.
- Accuracy alone is misleading: 78 % of the wavs are normal, so a model that says "normal" more often gets a higher
  accuracy while it misses more sick hearts.
- The official model of the platform: the highest mean validation recall, then F1 (PDF) — see the README benchmark.

## 2. The final result, model by model (validation: mean of the 3 folds of the 80 % train part)

### SVM (notebook 02) — winner: Zaineb's 72 features

| Candidate | From | Recall | F1 | Precision | Accuracy | Kept |
|---|---|---|---|---|---|---|
| Mel spectrogram flattened, rbf (Farah's original, 2 s) | Farah | 0.840 | 0.639 | 0.516 | — | |
| 72 features, rbf, default C/gamma | Zaineb | **0.894** | **0.705** | 0.582 | 0.832 | ✓ |
| 72 features, rbf, C/gamma tuned with F2 inside each fold | merged | 0.894 | 0.705 | 0.582 | 0.832 | (same) |
| 72 features + Farah's Mel features side by side | merged | 0.845 | 0.676 | 0.564 | 0.818 | |

Zaineb's 72 features (duration, SNR, mean/std of 20 Mel bands, mean/std of 13 MFCC, spectral centroid, ZCR, on the
whole wav) beat Farah's flattened Mel spectrogram. Tuning C and gamma with the F2 score (recall counts twice) inside
each fold chose the default values (C = 1, gamma = "scale"), so the default SVM stays. Putting both feature sets side
by side did worse (the 480 Mel numbers drown the 72 summaries). Notebook 02 keeps Farah's study of the Mel features
(durations 2 / 5 / 10 s, linear vs rbf) as data mining.

### Decision tree (notebook 03) — winner: MERGED (Zaineb's features and idea + the project's recall rule)

| Candidate | From | Recall | F1 | Precision | Accuracy | Kept |
|---|---|---|---|---|---|---|
| 26 MFCC, gini, depth 10 (Farah's original) | Farah | 0.688 | 0.624 | 0.571 | 0.814 | |
| 72 features, depth chosen inside each fold by F1 (Zaineb's method) | Zaineb | 0.770 | 0.651 | 0.570 | 0.815 | |
| 26 MFCC, settings by the F1 rule (entropy, depth 10) | merged | 0.737 | 0.654 | 0.588 | 0.825 | |
| 26 MFCC, settings by the recall rule (entropy, depth 9) | merged | 0.767 | 0.650 | 0.565 | 0.814 | |
| 72 features, settings by the F1 rule (gini, depth 8) — the first version of notebook 03 | merged | 0.778 | **0.660** | 0.574 | 0.820 | |
| **72 features, settings by the recall rule (entropy, depth 7)** | merged | **0.820** | 0.657 | 0.550 | 0.807 | ✓ |

The criterion and the depth are chosen with the folds of the train part (Zaineb's idea), and **by recall first**:
among the settings that lose at most 0.01 of F1 compared with the best F1, the highest recall. It finds 82 % of the
sick hearts instead of 78 % for almost the same F1.

### CNN (notebook 04) — winner: Farah's CNN

| Candidate | From | Recall | F1 | Precision | Accuracy | Kept |
|---|---|---|---|---|---|---|
| **Conv 16/32/64 + MaxPooling, Flatten, Dense 32, Dropout 0.5, 20 epochs** | Farah | **0.938** | 0.628 | 0.472 | 0.750 | ✓ |
| 64 × 79 spectrogram, BatchNorm + GlobalAveragePooling + softmax, early stopping | Zaineb | 0.668 | **0.690** | 0.715 | **0.865** | trade-off |
| Farah's blocks + BatchNorm + class_weight + early stopping (10 %), Flatten | merged | 0.534 | 0.411 | 0.364 | 0.773 | |
| the same with GlobalAveragePooling | merged | 0.692 | 0.660 | 0.639 | 0.838 | trade-off |

**Trade-off written down:** Zaineb's CNN (and the merged CNN with GlobalAveragePooling) have a higher F1, precision
and accuracy, but they find only 67–69 % of the sick hearts against 94 % for Farah's CNN. The PDF asks for the model
that maximizes the recall, so Farah's CNN stays. (With early stopping on 10 % of the fold, the BatchNorm networks
stopped early and became less sensitive to the abnormal class.)

### Transfer learning ResNet50 (notebook 06) — winner: MERGED (Farah's classifier + Zaineb's early stopping and size)

| Candidate | From | Recall | F1 | Precision | Accuracy | Kept |
|---|---|---|---|---|---|---|
| BatchNorm + Dense 128 + Dropout 0.5, 30 epochs (magma image 224 × 224) | Farah | 0.789 | **0.748** | 0.713 | 0.881 | |
| grey image 128 × 128, Dense 256, early stopping (part 1, other folds) | Zaineb | 0.757 | 0.740 | 0.723 | 0.881 | |
| Farah's classifier + early stopping (10 % of the fold) | merged | 0.780 | 0.729 | 0.687 | 0.870 | |
| **Farah's classifier with Dense 256 + early stopping** | merged | **0.793** | 0.746 | 0.704 | 0.879 | ✓ |

A little more recall for the same F1 (−0.002). The same method is used for VGG16 (kept in MLflow to compare).

### Transformer-CNN (notebook 07)

Model 5 is new (none of us had it): no merge.

### Ensemble (notebook 08) — merging the models themselves

Soft vote = the mean of the probabilities of "abnormal" of several models. The validation probabilities are
"out-of-fold" (each wav of the 80 % train part is scored by the model of the fold where it was in the validation
third, the same 3 folds for every model), so the combination is chosen without the test set. The SVM is not in the
vote: it gives a distance to its border, not a probability.

| Members (soft vote) | Recall | F1 | Precision | Accuracy | AUC | Kept |
|---|---|---|---|---|---|---|
| CNN alone (to compare) | 0.938 | 0.628 | 0.472 | 0.750 | 0.905 | |
| CNN + ResNet50 | 0.847 | **0.732** | 0.645 | 0.861 | 0.942 | |
| CNN + Transformer-CNN | **0.867** | 0.660 | 0.553 | 0.797 | 0.925 | |
| CNN + ResNet50 + Transformer-CNN | 0.863 | 0.712 | 0.616 | 0.841 | 0.944 | |
| **CNN + ResNet50 + Transformer-CNN + decision tree** | 0.858 | 0.714 | 0.619 | 0.843 | 0.941 | ✓ |

Applied in order with the rule: CNN + Transformer-CNN has more recall than CNN + ResNet50; adding ResNet50, then the
tree, each time gives a higher F1 for less than 0.01 of recall. The ensemble finds fewer sick hearts than the CNN alone
(0.858 vs 0.938) but makes far fewer false alarms (precision 0.619 vs 0.472): it is the 6th line of the benchmark,
not the official answer (the official model is the highest validation recall: the CNN). Test (once, at the end):
recall 0.933, F1 0.726, AUC 0.954.

## 3. Part 1: the first comparison (the first protocol: 3 folds on all the 3681 wavs)

Before the fixed test set existed, Zaineb's 4 methods were run on Farah's first protocol. Same conclusions: her 72
features won for the SVM (recall 0.887 vs 0.829) and the tree (0.863 vs 0.718), Farah's CNN (0.925 vs 0.798) and
ResNet50 (0.800 vs 0.757) won. Zaineb's own experiment showed that grouping the folds by patient changes the SVM F1
by only 0.005; only PASCAL names give the patient, so the prof's `KFold` is kept for every model (notebook 01 shows
the patient overlap for the PASCAL wavs).

## 4. Inventory against the PDF deliverables

| What the PDF asks | Farah had | Zaineb had | Decision |
|---|---|---|---|
| `01_eda_data_wrangling.ipynb` | fusion, durations, SNR, FFT, Mel, KDE, ZCR | the same starting notebook, no ZCR, no cleaning | Farah's + cleaning + the fixed test set + Zaineb's patient-overlap check |
| Model 1: decision tree (1D MFCC) | 26 MFCC, depth chosen on a split with the test wavs | 72 features (incl. the MFCC), depth by CV on the train part | **merged** (see 2) |
| Model 2: SVM rbf | flattened Mel, duration study | 72 features, GridSearch | **Zaineb's features** (see 2) |
| Model 3: custom CNN | Conv 16/32/64 | BatchNorm + GAP + softmax (PyTorch) | **Farah's** (see 2) |
| Model 4: pre-trained CNN | ResNet50 + VGG16, magma 224 | ResNet50 grey 128 | **merged** (see 2) + Zaineb's idea of the image preparation inside the model (ONNX export, notebook 05) |
| Model 5: Transformer-CNN | — | — | new notebook 07 |
| MLflow | inline logging, 1 nested run per fold | `mlflow_helper.py`, 1 registry entry per model | Farah's (shows the folds, simpler) |
| ONNX + `cardio_model.onnx` | notebook 05 | none (models served from MLflow with PyTorch) | Farah's |
| Services | audio engine (ONNX) + thin gateway | gateway that loads models from MLflow with PyTorch | Farah's (follows the PDF) |
| `docker-compose.yml` | none | 2 services | new, 6 services |
| `tests/` | none | `test_cnn_inference.py` (generated wav, mocks) | ideas reused in `tests/` |
| smoke test with real wavs | none | `check_api.py` | adapted as `scripts/check_platform.py` (test wavs only) |

## 5. What came from Zaineb (credits)

| Zaineb's piece | Where it is now |
|---|---|
| The 72-feature extraction | notebooks 02 and 03, audio engine `features_extended()` |
| Settings chosen by cross-validation on the train part only | notebook 03 (with the recall rule) |
| Early stopping + Dense 256 for the transfer-learning classifier | notebook 06 |
| Image preparation inside the transfer-learning model | ONNX export of ResNet50, notebook 05 |
| Patient-leakage awareness | the patient-overlap check in notebook 01 |
| Test style (wav generated with soundfile, mocked services) | `tests/` |
| Smoke test of the API with real wavs | `scripts/check_platform.py` |
