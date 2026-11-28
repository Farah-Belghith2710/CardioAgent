# CHANGES_FOR_FARAH — what changed, in simple words

This file is for you (Farah): what was done on the branch `final-merge`, why, and what the prof may ask.
The full technical report is the [`README.md`](README.md); the merge with Zaineb is in [`MERGE_NOTES.md`](MERGE_NOTES.md).

## 1. The data and the protocol (notebook 01)

- 85 broken wavs are removed (too short, silent, saturated): **3681 wavs** stay (2855 normal, 826 abnormal).
- A **fixed test set** of 20 % (737 wavs, 165 abnormal) is saved in `splits/test_files.json`. No notebook uses it to
  train or to choose anything. It is used **once**, at the end, for each final model.
- The other 80 % (2944 wavs) is cut in **3 folds** (`KFold(3, shuffle=True, random_state=1)`, like the TP): this is
  the **validation**. Every model uses the same 3 folds, so the comparisons are fair.

## 2. The merge with Zaineb (notebook 09 + the first cell of 02, 03, 04, 06)

For each model, all the versions (yours, Zaineb's, mixes of the two) were run on the same folds, and the rule was:
**keep the higher validation recall**, or a higher F1 if the recall does not drop by more than 0.01.

| Model | Winner | Why |
|---|---|---|
| SVM | Zaineb's 72 features | recall 0.894 vs 0.840 with your Mel features |
| Decision tree | mix: Zaineb's features + depth chosen in the folds by recall | recall 0.820 vs 0.688 for your first tree |
| CNN | **yours** | recall 0.938; Zaineb's CNN has a better F1 but finds only 67 % of the sick hearts |
| ResNet50 | mix: your classifier + Zaineb's early stopping and Dense 256 | recall 0.793 vs 0.789, same F1 |

## 3. The new things

- **Transformer-CNN** (notebook 07, model 5 of the PDF): small CNN + 2 Transformer encoder blocks (self-attention
  between the moments of the heartbeat). An AST is too big for our CPU.
- **Ensemble** (notebook 08): the mean of the probabilities of CNN + ResNet50 + Transformer-CNN + tree. Best test
  recall (0.933) with a better precision than the CNN.
- **Every notebook** shows the validation of each fold, the mean of the 3 folds, the test result, and the training /
  validation curves, and logs all of it in MLflow (`audio_benchmark_final`, 1 run per model + 1 small run per fold).
- **Notebook 05**: exports the 5 models to ONNX, checks that ONNX gives the same answers, makes `cardio_model.onnx`
  (the CNN = the best validation recall) and the Model Registry `cardio_model@champion`.
- **The platform** (6 Docker containers, `docker compose up -d --build`): LLM (Ollama), Qdrant (RAG), audio engine,
  agents (LangGraph), API gateway (web page on http://localhost:8000), MLflow.
- **RAG**: 17 documents (French cardiology courses, the HAL summary of the ESC 2022 recommendations, the PhysioNet
  2016 papers) → 509 pieces in Qdrant. The 2025 ESC guidelines are **not** used: the ESC forbids using them in AI
  tools without a licence.
- **3 agents**: Analyste (reads the numbers) → Cardiologue (explains with the RAG documents, with citations) →
  Auditeur (checks: no diagnosis, no prescription, sources, a doctor in the disclaimer). Pydantic checks the JSON.
- **Fine-tuning QLoRA** (Google Colab, T4 GPU): Qwen 2.5 7B on 506 French examples → `cardio-llm` (GGUF 4-bit).
- **Tests**: 116 tests, 91 % coverage, GitHub Actions at each push (no key needed).

## 3b. The chat with the assistant (branch `chat-feature`)

- **New page** (http://localhost:8000): white and blue, the analysis on the left, a chat on the right. The audio
  result appears in ~1 second; the agents' report arrives a few minutes later; the chat works meanwhile.
- **You can ask** about your wav ("Dois-je voir un médecin ?", "Que signifie la probabilité ?") or about the heart
  ("C'est quoi un souffle ?"). Off-topic questions ("comment faire une pizza ?") get a polite refusal.
- **The Auditeur was fixed**: the small LLM now answers simple true/false questions and the code decides, so good
  reports are approved instead of being replaced by the backup report.

## 3c. The online LLM with an API key (branch `online-llm`)

- **Every day (online mode)**: your key goes in the file `.env` (`LLM_API_KEY=...`), never in a chat or a commit.
  Start with `powershell -ExecutionPolicy Bypass -File scripts\start_online_mode.ps1`. Tested with your Groq key
  (model `openai/gpt-oss-120b`): a report in ~10 s and a chat answer in ~1.5 s, instead of 2–7 min and 1–2 min with
  the local LLM (and the much bigger online model writes reports that the Auditeur accepts more often than the local 3B). Groq keys start with `gsk_`, xAI keys with `xai-`: the code
  finds the provider alone.
- **Demo for the prof (local mode, the version of the PDF)**:
  `powershell -ExecutionPolicy Bypass -File scripts\start_local_mode.ps1` starts the 6 containers with Ollama:
  nothing leaves the PC (slower). Back with `scripts\start_online_mode.ps1`. No file to edit.
- The footer of the page says « Mode en ligne » or « Mode local ».
- If the key is wrong or there is no internet: the report is the safe one written by the code, and the chat says
  « Le service d'IA en ligne n'est pas disponible pour le moment (vérifiez la clé dans .env). ».

## 3d. The 503 of the audio engine is fixed + the dark page (28/11)

- **What was wrong**: the audio engine warmed up its 5 models *before* opening its port, so on a busy PC (and with
  `llm-server` started by Docker Desktop's ▶ button) it stayed deaf for minutes: « Service injoignable » (503).
- **Now**: it opens its port at once and warms up in the background (`/health` says `warming_up`); each model uses
  2 CPU threads; the page says « Le moteur audio démarre… Réessayez dans quelques secondes » instead of a Python
  error. A cold start takes ~55 s instead of minutes; the scripts no longer rebuild the images (`-Build` if you
  change the code).
- **Use the scripts** (not the ▶ of the whole stack in Docker Desktop): in the online mode, `llm-server` must not run.
- **The page**: dark navy theme with a soft blue glow, the heart icon in the browser tab, the title
  « CardioAgent-Ops », no subtitle, and the status « ● Mode en ligne » fully visible.

## 4. Questions the prof may ask (and short answers)

- **Why is the CNN the official model when its F1 is the lowest?** The PDF asks to maximize the recall of abnormal:
  missing a sick heart is worse than a false alarm. The CNN finds 94 % of the sick hearts on validation. The other
  models are still shown in the benchmark of every answer, and the ensemble is the compromise.
- **Why not choose with the test set?** Then the test result is not an honest estimate any more (we would have
  chosen the model that is lucky on those 737 wavs). Everything is chosen on the 3 validation folds; the test is
  used once.
- **Why is accuracy misleading?** 78 % of the wavs are normal: a model that always says "normal" gets 78 % accuracy
  and finds 0 sick hearts.
- **Why 3 folds and not grouped by patient?** It is the protocol of the TP. Only the PASCAL file names give the
  patient (10.7 % of the wavs); Zaineb's test showed that grouping changes the SVM F1 by only 0.005 (notebook 01 and
  MERGE_NOTES).
- **Why ONNX?** One small file per model, the container runs it with ONNX Runtime only (no TensorFlow, no
  scikit-learn), and the notebook checks that the answers are the same.
- **What does the ensemble add?** Different models make different mistakes; the mean of their probabilities finds
  93 % of the sick hearts on the test set with fewer false alarms than the CNN alone.
- **How do the agents avoid inventing things?** The Cardiologue can only use the RAG passages and must cite them; the
  Auditeur checks the report (rules + LLM); the numbers come from the audio engine, not from the LLM; if the report
  is still not approved after 1 retry, the code writes a safe report.
- **How does the off-topic filter work?** First lists of keywords (cœur, souffle, médecin, wav, résultat… → yes;
  pizza, recette, football, code… → no), so it is fast and always gives the same answer. Only unclear questions are
  sent to the LLM with a yes/no question. The refusal text is written in the code, never by the LLM.
- **Why can't the chat invent numbers?** The numbers come from the analysis JSON sent by the page; the code gives
  them to the LLM, and if the answer contains a decimal number that is not in the analysis, the code replaces the
  answer by a safe one written from the analysis.
- **Why does it never give a diagnosis?** The prompt forbids it, and the code checks the answer with the Auditeur's
  list of forbidden words (diagnosis, drugs, doses); a refused answer is replaced, and every answer says that only a
  doctor can confirm (and gives the emergency number for chest pain or fainting).
- **Where are the data protected ("sans fuite de données")?** In the local mode everything runs in our Docker
  network (no external API). In both modes the LLM never receives the audio or the file name, and MLflow stores only
  numbers (never the key).
- **Why an online LLM?** On our PC without a usable GPU, the local LLM needs minutes per report; Groq / xAI answer in
  seconds, which is better for a demo. The local version is still there, in one command.
- **What leaves the PC in the online mode?** Only the question and the numbers of the analysis (prediction,
  probabilities, scores of the models), sent to the provider. Never the audio, never the file name; the page footer
  says it.
- **How do you show the local version?** `scripts\start_local_mode.ps1`: the same code with `LLM_PROVIDER=ollama`, the
  6 containers of the PDF, and the footer says « Mode local : rien ne quitte votre ordinateur. »

## 5. What you still do by hand (if you want)

- `scripts\start_online_mode.ps1` (or `scripts\start_local_mode.ps1`), then open http://localhost:8000 and drop a
  wav of the test set.
- To use the fine-tuned LLM: put `cardio-qwen2.5-7b-q4_k_m.gguf` in `llm_server/models/` and start
  `scripts\start_local_mode.ps1 -Model cardio-llm` (see the README, sections 2b and 6).
