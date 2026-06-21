#!/usr/bin/env bash
# Longitude-band extraction for the OOM states TX(48) then CA(06), one state at a
# time (each band parse uses 15-40GB). Resumable; bands that OOM auto-split.
set -u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate street
cd /home/wnlab/CK_street
LOGDIR=data/outputs/state_logs; mkdir -p "$LOGDIR"
ts() { date -u +%FT%TZ; }
FLOG="$LOGDIR/_strips.log"
echo "=== STRIPS ORCH START $(ts) (pid $$) ===" >> "$FLOG"
for s in 48 06; do
  log="$LOGDIR/state_${s}_strips.log"
  echo "[$(ts)] $s START" >> "$FLOG"
  echo "=== STRIPS START $s $(ts) ===" >> "$log"
  python 01d_extract_strips.py --state "$s" --per-band 1000 >> "$log" 2>&1
  echo "=== STRIPS END $s rc=$? $(ts) ===" >> "$log"
  echo "[$(ts)] $s END" >> "$FLOG"
done
echo "=== STRIPS ORCH COMPLETE $(ts) ===" >> "$FLOG"
