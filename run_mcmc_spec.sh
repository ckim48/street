#!/usr/bin/env bash
# Detached, resumable driver for the Stage-5 spec MCMC optimal-network search
# over the top-1000 UOI tracts. Resumes (skips tracts already in summary.json).
#
#   ./run_mcmc_spec.sh [TOP] [ITERS] [WEIGHTS] [REPLICAS] [TEMPS] [PROCS]
# defaults below = the "balanced" config (~13h for the full top-1000 on 24 procs)
set -euo pipefail
cd "$(dirname "$0")"

TOP=${1:-1000}
ITERS=${2:-4000}
WEIGHTS=${3:-2}
REPLICAS=${4:-2}
TEMPS=${5:-4}
PROCS=${6:-24}

source ~/anaconda3/etc/profile.d/conda.sh && conda activate street
LOG=data/outputs/sampler_spec/run.log
mkdir -p data/outputs/sampler_spec

echo "=== $(date) launching: top=$TOP iters=$ITERS w=$WEIGHTS r=$REPLICAS temps=$TEMPS procs=$PROCS ===" | tee -a "$LOG"
python 05_mcmc_spec.py --top "$TOP" --iters "$ITERS" --weights "$WEIGHTS" \
    --replicas "$REPLICAS" --temps "$TEMPS" --procs "$PROCS" --resume 2>&1 | tee -a "$LOG"
echo "=== $(date) DONE ===" | tee -a "$LOG"
