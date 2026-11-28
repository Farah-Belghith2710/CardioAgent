# Online mode (the default): the LLM of the agents and of the chat is an online API (Groq or xAI) with the key of .env.
# Fast (a few seconds), but the question and the numbers of the analysis are sent to the provider
# (never the audio, never the file name). 5 containers: llm-server (Ollama) is stopped, not deleted.
#
#   powershell -ExecutionPolicy Bypass -File scripts\start_online_mode.ps1           (every day: fast)
#   powershell -ExecutionPolicy Bypass -File scripts\start_online_mode.ps1 -Build    (after a change of the code)
#
# Back to the local mode (the version of the PDF): scripts\start_local_mode.ps1
param([switch]$Build)

Set-Location (Split-Path $PSScriptRoot -Parent)    # the folder of docker-compose.yml and .env

if (-not (Test-Path .env)) {
    Write-Host "No file .env: copy .env.example to .env and paste your key (LLM_API_KEY=...)"
}

# remove the values set by the local mode in this terminal: the settings of .env are used again
Remove-Item Env:LLM_PROVIDER -ErrorAction SilentlyContinue
Remove-Item Env:LLM_MODEL -ErrorAction SilentlyContinue

docker compose --profile local-llm stop llm-server    # the local LLM is stopped (kept, with its model)
# --build only when asked: rebuilding the images takes ~3 minutes on our PC, even when nothing changed
if ($Build) { docker compose up -d --build } else { docker compose up -d }
docker compose ps
Write-Host "Online mode started: http://localhost:8000 (the footer of the page says 'Mode en ligne')."
