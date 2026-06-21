#!/usr/bin/env bash
# Retry the 5 states that GIVEUP'd in the parallel run.
# Phase A: VA(51, was a 502 download blip) + NC(37, mega-county tail) in parallel.
# Phase B: TX(48) IL(17) CA(06) ONE AT A TIME so each mega-county build gets the
#          full ~60GB. Resumable: done counties/tracts are skipped, so only the
#          undone (mostly large) counties are rebuilt. The truly oversized counties
#          (LA 06037 / Cook 17031 / Harris 48201 etc.) may still OOM and be left as
#          gaps to backfill later with sub-county chunking.
set -u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate street
cd /home/wnlab/CK_street
LOGDIR=data/outputs/state_logs
mkdir -p "$LOGDIR"

do_state() {
  s="$1"; tries="$2"
  log="data/outputs/state_logs/state_${s}.log"
  echo "=== RETRY START $s $(date -u +%FT%TZ) ===" >> "$log"
  for attempt in $(seq 1 "$tries"); do
    if python 01_extract_networks_pbf.py --state "$s" >> "$log" 2>&1; then
      echo "=== DONE $s (retry attempt $attempt) $(date -u +%FT%TZ) ===" >> "$log"
      return 0
    fi
    echo "=== FAIL $s retry attempt $attempt $(date -u +%FT%TZ) ===" >> "$log"
    sleep 30
  done
  echo "=== GIVEUP $s (after retry) $(date -u +%FT%TZ) ===" >> "$log"
}
export -f do_state

echo "=== RETRY ORCH START $(date -u +%FT%TZ) ===" >> "$LOGDIR/_orchestrator.log"
# Phase A: small/transient failures, 2 in parallel, more attempts (download retry)
printf '%s\n' "51 5" "37 3" | xargs -n2 -P2 bash -c 'do_state "$@"' _
# Phase B: mega states, strictly one at a time, fewer attempts (OOM won't fix itself)
for s in 48 17 06; do do_state "$s" 2; done
echo "=== RETRY ORCH COMPLETE $(date -u +%FT%TZ) ===" >> "$LOGDIR/_orchestrator.log"
