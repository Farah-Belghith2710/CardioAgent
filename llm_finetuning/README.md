# llm_finetuning/ — QLoRA fine-tuning of the cardiology LLM

| File | Role |
|---|---|
| `build_dataset.py` | Writes `cardio_instructions.jsonl`: 506 French instruction examples (explain an audio-engine result, heart-sound basics, when to see a doctor, safety rules). No patient data, fixed seed. |
| `cardio_instructions.jsonl` | The dataset (`instruction` / `input` / `output`). |
| `train_qlora_colab.ipynb` | Google Colab notebook (free T4 GPU): Unsloth + TRL `SFTTrainer`, base `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`, LoRA r=16, alpha=16, lr 2e-4, 1 epoch. Saves the loss history, the LoRA adapter and a **GGUF q4_k_m** export. |
| `log_finetuning_to_mlflow.py` | Run on the PC after Colab: loss curve (1 point per step), params, peak VRAM, artifacts → MLflow experiment `llm_finetuning`; registers `cardio_llm` (the adapter). |
| `Modelfile` | Ollama recipe of `cardio-llm` (the GGUF + Qwen ChatML template + French system prompt, temperature 0.2, num_ctx 4096). |

## Steps
1. `venv\Scripts\python llm_finetuning\build_dataset.py`
2. Colab: upload `train_qlora_colab.ipynb` + `cardio_instructions.jsonl`, Runtime → T4 GPU, Run all (≈ 40–60 min).
3. From Google Drive `MyDrive/cardioagent_llm/`: `trainer_state.json`, `run_info.json` and the unzipped adapter →
   `llm_finetuning/outputs/`; the `.gguf` → `llm_server/models/` (both folders are git-ignored: big files).
4. `venv\Scripts\python llm_finetuning\log_finetuning_to_mlflow.py`
5. `docker compose up -d --build llm-server`: the server sees the GGUF and runs `ollama create cardio-llm`;
   start the local mode with it: `scripts\start_local_mode.ps1 -Model cardio-llm` (see the main README, section 2b).

Until the GGUF exists, the platform runs the base model `qwen2.5:3b-instruct` pulled by Ollama (the default
`LLM_MODEL` of `docker-compose.yml`: on this PC the LLM runs on the CPU, and Docker has ~7.7 GB of memory).
The fine-tuned model is a 7B (~4.7 GB): with `-Model cardio-llm` the reports are better but slower on a CPU.
