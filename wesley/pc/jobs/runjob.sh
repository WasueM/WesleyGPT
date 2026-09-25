#!/bin/bash
# Wesley wrote this
# Runs ~/jobs/current.sh under a live wsl.exe client (started by the Windows
# Scheduled Task "nanochat-job"), so WSL never idle-shuts-down mid-run.
# Output: ~/jobs/logs/<name>.log ; status: ~/jobs/status
mkdir -p ~/jobs/logs
JOB=~/jobs/current.sh
[ -f "$JOB" ] || { echo "no job $(date)" > ~/jobs/status; exit 0; }
NAME=$(grep -m1 '^# name:' "$JOB" | cut -d: -f2 | tr -d ' ')
NAME=${NAME:-job}
echo "running $NAME since $(date '+%F %T')" > ~/jobs/status
bash "$JOB" >> ~/jobs/logs/$NAME.log 2>&1
rc=$?
echo "finished $NAME rc=$rc at $(date '+%F %T')" > ~/jobs/status
