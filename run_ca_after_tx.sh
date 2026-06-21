#!/usr/bin/env bash
# Wait for the currently-running TX(48) batched extraction to finish, then run
# CA(06) — they must not overlap (each batch parse uses 20-50GB). CA was skipped
# when its orchestrator got killed during a restart; this backfills it.
set -u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate street
cd /home/wnlab/CK_street
LOGDIR=data/outputs/state_logs
ts() { date -u +%FT%TZ; }
echo "[$(ts)] CA-after-TX waiter started (pid $$)" >> "$LOGDIR/_batched.log"
while pgrep -f "01c_extract_batched.py --state 48" >/dev/null 2>&1; do sleep 60; done
echo "[$(ts)] TX gone -> starting CA(06)" >> "$LOGDIR/_batched.log"
log="$LOGDIR/state_06_batched.log"
echo "=== BATCHED START 06 (after TX) $(ts) ===" >> "$log"
python 01c_extract_batched.py --state 06 --cap 1200 >> "$log" 2>&1
echo "=== BATCHED END 06 rc=$? $(ts) ===" >> "$log"
echo "[$(ts)] 06 done" >> "$LOGDIR/_batched.log"
