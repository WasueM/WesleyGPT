# Wesley wrote this
# name: base-chateval
cd ~/nanochat && source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
echo "===== $(date '+%F %T') chat_eval on the PRETRAINED (base) d12"
python -m scripts.chat_eval -i base -g d12
echo "===== $(date '+%F %T') done"
