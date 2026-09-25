#!/bin/bash
# Wesley wrote this
cd ~/nanochat && source .venv/bin/activate
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
exec python wesley_complete.py "$@"
