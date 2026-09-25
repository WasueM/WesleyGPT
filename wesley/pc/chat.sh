#!/bin/bash
# Wesley wrote this
# Chat with the d12 nanochat model. From the Mac:  ssh -t wesle@supercomputer wsl -e bash /home/wesley/chat.sh
cd ~/nanochat && source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
exec python -m scripts.chat_cli -i sft -g d12 "$@"
