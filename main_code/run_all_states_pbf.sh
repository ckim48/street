#!/usr/bin/env bash
# Full-U.S. UOI run via Geofabrik .pbf: Stage 1 (per-state pbf extraction with
# simplify + fast truncation) then Stage 2 (UOI). Sequential by state,
# resumable (done tracts skipped, cached .pbf reused), retries each state 2x.
# Designed to run detached via nohup for ~1-2 days.
set -u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate street
cd /home/wnlab/CK_street

LOG=data/outputs/run_all_states_pbf.log
mkdir -p data/outputs

# 50 states + DC (11) + Puerto Rico (72)
STATES="01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 \
27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 \
54 55 56 72"

ts() { date -u +%FT%TZ; }
echo "=== [$(ts)] PBF RUN START (pid $$) ===" >> "$LOG"

for s in $STATES; do
  if grep -q "STATE $s DONE" "$LOG" 2>/dev/null; then
    echo "=== [$(ts)] STATE $s already DONE, skip ===" >> "$LOG"; continue
  fi
  echo "=== [$(ts)] STATE $s START ===" >> "$LOG"
  for attempt in 1 2; do
    if python 01_extract_networks_pbf.py --state "$s" >> "$LOG" 2>&1; then
      echo "=== [$(ts)] STATE $s DONE (attempt $attempt) ===" >> "$LOG"; break
    fi
    echo "=== [$(ts)] STATE $s FAIL attempt $attempt, sleep 60s ===" >> "$LOG"
    sleep 60
  done
done
echo "=== [$(ts)] STAGE 1 COMPLETE ===" >> "$LOG"

echo "=== [$(ts)] STAGE 2 START ===" >> "$LOG"
python 02_compute_uoi.py >> "$LOG" 2>&1
echo "=== [$(ts)] ALL COMPLETE ===" >> "$LOG"
