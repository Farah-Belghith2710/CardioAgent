# llm_server/ — files of Service 1 (llm-server, Ollama)

Put the fine-tuned model made on Colab (`llm_finetuning/train_qlora_colab.ipynb`) here:

    llm_server/models/cardio-qwen2.5-7b-q4_k_m.gguf      (~4.7 GB, git-ignored)

At start, the container `llm-server` (Dockerfile: `docker/llm-server.Dockerfile`, start script:
`docker/llm-server-entrypoint.sh`) sees the file and runs `ollama create cardio-llm` with
`llm_finetuning/Modelfile`. The local mode with it: `scripts\start_local_mode.ps1 -Model cardio-llm` (main README,
section 2b): the agents use the fine-tuned model. The `llm-server` service is in the compose profile `local-llm`.

Without the GGUF, the platform uses the base model `qwen2.5:3b-instruct` (or the one set in `LLM_MODEL`),
downloaded once by Ollama and kept in the Docker volume `ollama-data`.
