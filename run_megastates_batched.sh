#!/usr/bin/env bash
# Regional-batch extraction for the OOM-prone mega states CA(06) and TX(48),
# run ONE STATE AT A TIME (each batch can use ~30-50GB). 01c packs counties into
# RAM-sized batches, runs each in its own subprocess (OOM-isolated), and tiles
# any single county too big to parse whole. Resumable. IL(17) already done via
# 01b; CT(09) handled separately (pyrosm bbox bug).
set -u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate street
cd /home/wnlab/CK_street
LOGDIR=data/outputs/state_logs; mkdir -p "$LOGDIR"
ts() { date -u +%FT%TZ; }
FLOG="$LOGDIR/_batched.log"
echo "=== BATCHED ORCH START $(ts) (pid $$) ===" >> "$FLOG"
for s in 06 48; do
  log="$LOGDIR/state_${s}_batched.log"
  echo "=== BATCHED START $s $(ts) ===" >> "$log"
  echo "[$(ts)] $s START" >> "$FLOG"
  python 01c_extract_batched.py --state "$s" --cap 1200 >> "$log" 2>&1
  rc=$?
  echo "=== BATCHED END $s rc=$rc $(ts) ===" >> "$log"
  echo "[$(ts)] $s END rc=$rc" >> "$FLOG"
done
echo "=== BATCHED ORCH COMPLETE $(ts) ===" >> "$FLOG"
