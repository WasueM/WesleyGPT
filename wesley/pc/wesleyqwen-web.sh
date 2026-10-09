#!/bin/bash
# Wesley wrote this
# Browser chat with base Qwen3.5-2B and the WesleyQwen variants, with a video picker. From the Mac run `qwen-web`
# (opens the tunnel and the browser), or by hand:  ssh -t -L 7860:127.0.0.1:7860 wesle@supercomputer wsl -e bash /home/wesley/wesleyqwen-web.sh
cd ~/wesleyqwen
export WESLEYQWEN_BASE="$HOME/wesleyqwen/base/Qwen3.5-2B" HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error
exec .venv/bin/python -m wesleyqwen.web "$@" 2> >(grep -vE "Python version|Skipping import|causal_conv1d|Loading weights|cap_pixels_per_frame|warnings.warn" >&2)
