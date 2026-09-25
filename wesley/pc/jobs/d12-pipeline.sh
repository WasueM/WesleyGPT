# name: d12
# Wesley wrote this
# d12 practice run: data -> pretrain (auto-resume) -> base eval -> SFT -> chat eval.
# Re-runnable: each finished stage leaves a marker in ~/jobs/d12/, and pretraining
# resumes from the newest checkpoint, so a reboot only loses progress since the last save.
set -u
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
cd ~/nanochat && source .venv/bin/activate
M=~/jobs/d12; mkdir -p $M
CK=$NANOCHAT_BASE_DIR/base_checkpoints/d12
FINAL=2520
stage() { echo; echo "===== $(date '+%F %T') $1"; }
done_or_die() { if [ $1 -eq 0 ]; then touch $M/$2.done; else echo "!!!!! stage $2 failed rc=$1"; exit $1; fi; }

if [ ! -f $M/data.done ]; then stage "data: 60 shards"; python -m nanochat.dataset -n 60; done_or_die $? data; fi

if [ ! -f $M/pretrain.done ]; then
  LAST=$(ls $CK/model_*.pt 2>/dev/null | sed -E 's/.*model_0*([0-9]+)\.pt/\1/' | sort -n | tail -1)
  if [ "${LAST:-}" = "$FINAL" ]; then touch $M/pretrain.done
  else
    RESUME=""; [ -n "${LAST:-}" ] && RESUME="--resume-from-step=$LAST"
    stage "pretrain d12 ${RESUME:-from scratch}"
    python -m scripts.base_train --depth=12 --device-batch-size=8 --run=dummy --model-tag=d12 \
      --save-every=250 --eval-every=500 --eval-tokens=5242880 --core-metric-every=-1 --sample-every=500 $RESUME
    done_or_die $? pretrain
  fi
fi

if [ ! -f $M/base_eval.done ]; then stage "base eval (CORE, bpb, samples)"; python -m scripts.base_eval --model-tag=d12 --device-batch-size=8; done_or_die $? base_eval; fi
if [ ! -f $M/sft.done ]; then stage "SFT"; python -m scripts.chat_sft --model-tag=d12 --run=dummy; done_or_die $? sft; fi
if [ ! -f $M/chat_eval.done ]; then stage "chat eval"; python -m scripts.chat_eval -i sft -g d12; done_or_die $? chat_eval; fi
stage "ALL DONE. Talk to it: cd ~/nanochat && source .venv/bin/activate && python -m scripts.chat_cli"
