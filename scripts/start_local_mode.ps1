# Local mode = the version of the PDF: the 6 containers, the LLM runs in the container llm-server (Ollama) on this PC.
# Nothing leaves the computer, but the LLM is slow on a CPU (a report: several minutes, a chat answer: 1 to 3 min).
#
#   powershell -ExecutionPolicy Bypass -File scripts\start_local_mode.ps1                     (model qwen2.5:3b-instruct)
#   powershell -ExecutionPolicy Bypass -File scripts\start_local_mode.ps1 -Model cardio-llm   (our fine-tuned model)
#
# The first start downloads the model (~2 GB, kept in the Docker volume ollama-data) and loads it in memory.
# Back to the online mode: scripts\start_online_mode.ps1
param([string]$Model = "qwen2.5:3b-instruct")

Set-Location (Split-Path $PSScriptRoot -Parent)    # the folder of docker-compose.yml and .env

# these 2 values win over the file .env (only in this terminal): no file to edit
$env:LLM_PROVIDER = "ollama"
$env:LLM_MODEL = $Model

docker compose --profile local-llm up -d --build
docker compose --profile local-llm ps
Write-Host "Local mode started: http://localhost:8000 (the footer of the page says 'Mode local')."
Write-Host "The LLM can need several minutes to load the first time (docker compose ps: llm-server healthy)."
