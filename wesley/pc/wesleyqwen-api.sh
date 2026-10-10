#!/bin/bash
# Wesley wrote this
# The home-PC model API that MangumHub's /wesleygpt page reaches through Tailscale Funnel. Started at logon by the
# "wesleyqwen-api" Scheduled Task (wesley/pc/windows/register-wesleyqwen-api.ps1), which also keeps WSL awake.
# The key lives in ~/.wesleyqwen-api-key (mode 600); MangumHub holds the same value as Secret Manager's wesleyqwen-api-key.
cd ~/wesleyqwen
export WESLEYQWEN_BASE="$HOME/wesleyqwen/base/Qwen3.5-2B" WESLEYQWEN_GEMMA="$HOME/wesleyqwen/base/gemma-4-E2B-it" HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error
export WESLEYQWEN_API_KEY="$(cat ~/.wesleyqwen-api-key)"
export WESLEYQWEN_ALLOWED_ORIGINS="https://mangumhub.com,http://localhost:3000"
mkdir -p ~/wesleyqwen/logs
# Restart here rather than through the task's RestartCount: when the server died, the task stayed "running" and
# Windows never restarted it.
while true; do
  .venv/bin/python -u -m wesleyqwen.api "$@" >> ~/wesleyqwen/logs/api.log 2>&1
  echo "[api] exited with $?; restarting in 5 s" >> ~/wesleyqwen/logs/api.log
  sleep 5
done
