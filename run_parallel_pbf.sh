#!/usr/bin/env bash
# Parallel full-U.S. extraction for the remaining states (Stage 1).
# 6 concurrent state workers; big states are spaced out in the list so at most
# ~1-2 run at once (memory safety). Resumable: tracts with an existing GraphML
# are skipped, so already-DONE states are cheap no-ops. Per-state stdout logs
# live in data/outputs/state_logs/.
set -u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate street
cd /home/wnlab/CK_street

LOGDIR=data/outputs/state_logs
mkdir -p "$LOGDIR"
WORKERS=6

# remaining mid/small states interleaved with the 6 big ones (06 17 36 39 42 48)
STATES="29 30 31 06 32 33 17 34 35 36 37 38 39 40 41 42 44 45 48 46 47 49 50 51 53 54 55 56 72"

do_state() {
  s="$1"
  log="data/outputs/state_logs/state_${s}.log"
  echo "=== START $s $(date -u +%FT%TZ) ===" >> "$log"
  for attempt in 1 2 3; do
    if python 01_extract_networks_pbf.py --state "$s" >> "$log" 2>&1; then
      echo "=== DONE $s (attempt $attempt) $(date -u +%FT%TZ) ===" >> "$log"
      return 0
    fi
    echo "=== FAIL $s attempt $attempt $(date -u +%FT%TZ) ===" >> "$log"
    sleep 30
  done
  echo "=== GIVEUP $s $(date -u +%FT%TZ) ===" >> "$log"
}
export -f do_state

echo "=== ORCH START $(date -u +%FT%TZ) workers=$WORKERS ===" >> "$LOGDIR/_orchestrator.log"
printf '%s\n' $STATES | xargs -n1 -P"$WORKERS" bash -c 'do_state "$0"'
echo "=== ORCH ALL COMPLETE $(date -u +%FT%TZ) ===" >> "$LOGDIR/_orchestrator.log"
