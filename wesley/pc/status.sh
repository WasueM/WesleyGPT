# Wesley wrote this
uptime; cat ~/jobs/status; ls ~/jobs/d12/ 2>/dev/null
grep -E "=====|!!!!!|Traceback|Error" ~/jobs/logs/d12.log | tail -5
grep -E "^step" ~/jobs/logs/d12.log | tail -2 | cut -c1-200
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader
