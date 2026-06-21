#!/usr/bin/env bash
# Full-U.S. minus the 7 largest states (by tract count); resumable, retries 2x.
set -u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate street
cd /home/wnlab/CK_street
LOG=data/outputs/run_all_states_pbf.log
SKIP=" 06 48 36 12 42 17 39 "   # CA TX NY FL PA IL OH -> handle later
STATES="01 02 04 05 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 \
28 29 30 31 32 33 34 35 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56 72"
ts() { date -u +%FT%TZ; }
echo "=== [$(ts)] REST RUN START (pid $$, skip top7) ===" >> "$LOG"
for s in $STATES; do
  case "$SKIP" in *" $s "*) echo "=== [$(ts)] STATE $s SKIP (top7) ===" >> "$LOG"; continue;; esac
  if grep -q "STATE $s DONE" "$LOG" 2>/dev/null; then
    echo "=== [$(ts)] STATE $s already DONE, skip ===" >> "$LOG"; continue; fi
  echo "=== [$(ts)] STATE $s START ===" >> "$LOG"
  for attempt in 1 2; do
    if python 01_extract_networks_pbf.py --state "$s" >> "$LOG" 2>&1; then
      echo "=== [$(ts)] STATE $s DONE (attempt $attempt) ===" >> "$LOG"; break; fi
    echo "=== [$(ts)] STATE $s FAIL attempt $attempt, sleep 60s ===" >> "$LOG"; sleep 60
  done
done
echo "=== [$(ts)] REST STAGE 1 COMPLETE (top7 pending) ===" >> "$LOG"
