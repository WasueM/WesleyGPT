#!/bin/bash
# Wesley wrote this
# Chat with base Qwen3.5-2B and the WesleyQwen variants. From the Mac:  ssh -t wesle@supercomputer wsl -e bash /home/wesley/wesleyqwen-chat.sh
cd ~/wesleyqwen
export WESLEYQWEN_BASE="$HOME/wesleyqwen/base/Qwen3.5-2B" HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error
exec .venv/bin/python -m wesleyqwen.chat "$@" 2> >(grep -vE "Python version|Skipping import|causal_conv1d|Loading weights" >&2)
