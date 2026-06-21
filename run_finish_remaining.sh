#!/usr/bin/env bash
# Finish the remaining extraction: CT(09) small gap via county script, then the
# three mega states IL/CA/TX via 01b_extract_mega.py — ONE AT A TIME so each
# whole-state pbf parse gets the full ~60GB box (no OOM). All steps resumable:
# tracts with an existing GraphML are skipped.
set -u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate street
cd /home/wnlab/CK_street
LOGDIR=data/outputs/state_logs
mkdir -p "$LOGDIR"
ts() { date -u +%FT%TZ; }
FLOG="$LOGDIR/_finish.log"
echo "=== FINISH ORCH START $(ts) (pid $$) ===" >> "$FLOG"

run_county() {  # small-gap state, county-by-county script
  s="$1"; log="$LOGDIR/state_${s}.log"
  echo "=== COUNTY-FINISH START $s $(ts) ===" >> "$log"
  for a in 1 2 3; do
    if python 01_extract_networks_pbf.py --state "$s" >> "$log" 2>&1; then
      echo "=== COUNTY-FINISH DONE $s attempt $a $(ts) ===" >> "$log"
      echo "[$(ts)] $s DONE" >> "$FLOG"; return 0; fi
    echo "=== COUNTY-FINISH FAIL $s attempt $a $(ts) ===" >> "$log"; sleep 30
  done
  echo "[$(ts)] $s GIVEUP" >> "$FLOG"
}

run_mega() {  # mega state, one-parse-per-state script, strictly sequential
  s="$1"; log="$LOGDIR/state_${s}_mega.log"
  echo "=== MEGA START $s $(ts) ===" >> "$log"
  for a in 1 2; do
    if python 01b_extract_mega.py --state "$s" >> "$log" 2>&1; then
      echo "=== MEGA DONE $s attempt $a $(ts) ===" >> "$log"
      echo "[$(ts)] $s MEGA DONE" >> "$FLOG"; return 0; fi
    echo "=== MEGA FAIL $s attempt $a $(ts) ===" >> "$log"; sleep 30
  done
  echo "[$(ts)] $s MEGA GIVEUP" >> "$FLOG"
}

run_county 09          # CT, ~323 tracts (quick)
run_mega   17          # IL, ~3.1k  (smallest mega — validates 01b first)
run_mega   06          # CA, ~7.8k
run_mega   48          # TX, ~6.7k
echo "=== FINISH ORCH COMPLETE $(ts) ===" >> "$FLOG"
