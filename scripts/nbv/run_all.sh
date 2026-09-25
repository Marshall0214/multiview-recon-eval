#!/usr/bin/env bash
# M2 batch: 3 objects x 2 inits x 4 strategies x 2 seeds = 48 runs (x 5 trainings each).
# `ig` (v2) was added after the v1 failure analysis; it was developed on the chair only.
# Runs P jobs in parallel on one GPU (object-level scenes do not saturate a 4090).
# Timing fields in these runs are contaminated by parallelism; selection cost is measured separately.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=${P:-3}
OUT=${OUT:-/root/outputs/nbv}
DATA=${DATA:-/root/data/abo_render}
mkdir -p "$OUT"
for obj in B07J2YB486 B07B82PXCW B07HSCJZQM; do
  for init in biased uniform; do
    for seed in 0 1; do
      for strat in ${STRATS:-ours fps random ig}; do
        [ -f "$OUT/$obj/$init/$strat/seed$seed/round4.json" ] && continue   # resumable
        echo "$DATA/$obj $init $strat $seed"
      done
    done
  done
done | xargs -P "$P" -L 1 bash -c 'python scripts/nbv/run_nbv.py --scene $0 --init $1 --strategy $2 --seed $3 --out '"$OUT"' >> '"$OUT"'/log_$(basename $0)_$1_$2_$3.txt 2>&1 && echo "done $0 $1 $2 $3"'
