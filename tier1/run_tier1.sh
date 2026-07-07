#!/usr/bin/env bash
# Tier 1 end-to-end driver: HOLC boundary RD for the six deep-dive cities.
# Resumable-ish: each stage overwrites its own city outputs.
#
#   ./tier1/run_tier1.sh                 # all six cities, all stages
#   ./tier1/run_tier1.sh chicago detroit # subset
#   FETCH_CHRONEX=1 ./tier1/run_tier1.sh # also (re)download the 2.7 GB CHRONEX zip
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate street

CITIES=("$@")
[ ${#CITIES[@]} -eq 0 ] && CITIES=(chicago philadelphia baltimore detroit atlanta los_angeles)
echo "=== Tier 1 cities: ${CITIES[*]} ==="

# 0) data (HOLC/Markley/HISDAC always; CHRONEX only if FETCH_CHRONEX=1)
if [ "${FETCH_CHRONEX:-0}" = "1" ]; then bash tier1/00_fetch_tier1.sh --chronex
else bash tier1/00_fetch_tier1.sh; fi

cd tier1
python 01_holc_boundaries.py --cities "${CITIES[@]}"
python 02_decade_graphs.py   --cities "${CITIES[@]}"
python 03_compute_oi.py      --cities "${CITIES[@]}" --barriers street --max-seg 300
python 04_rd_estimate.py     --cities "${CITIES[@]}" --bandwidth 300
python 05_figures.py         --cities "${CITIES[@]}"
python 06_mechanism_validation.py     # mechanism set + 1940-vs-2020 same-place panels
echo "=== Tier 1 done -> results/tier1/ ==="
