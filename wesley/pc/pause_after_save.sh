# Wesley wrote this
# Pause d12 right after the step-750 checkpoint is fully written (step 751 appearing = save finished).
CK=~/.cache/nanochat/base_checkpoints/d12
until grep -q "^step 00751" ~/jobs/logs/d12.log; do sleep 5; done
ls -la $CK/*000750*
pkill -f "scripts.base_train" && echo "paused at $(date +%T)"
sleep 5; pgrep -f scripts.base_train || echo "base_train stopped"
cat ~/jobs/status
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
