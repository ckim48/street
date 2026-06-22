#!/usr/bin/env bash
# Final mega-state extraction: osmium-tile then per-tile whole-parse. TX then CA,
# one state at a time. Resumable; tile .pbf files cached under data/pbf/tiles.
set -u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate street
cd /home/wnlab/CK_street
LOGDIR=data/outputs/state_logs; mkdir -p "$LOGDIR"
ts() { date -u +%FT%TZ; }
FLOG="$LOGDIR/_osmium.log"
echo "=== OSMIUM ORCH START $(ts) (pid $$) ===" >> "$FLOG"
for s in 48 06; do
  log="$LOGDIR/state_${s}_osmium.log"
  echo "[$(ts)] $s START" >> "$FLOG"
  echo "=== OSMIUM START $s $(ts) ===" >> "$log"
  python 01e_extract_osmium.py --state "$s" --per-cell 600 >> "$log" 2>&1
  echo "=== OSMIUM END $s rc=$? $(ts) ===" >> "$log"
  echo "[$(ts)] $s END" >> "$FLOG"
done
echo "=== OSMIUM ORCH COMPLETE $(ts) ===" >> "$FLOG"
