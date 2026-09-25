#!/usr/bin/env bash
# M0: train gsplat `default` (3DGS) on Mip-NeRF 360 scenes and evaluate at 7k / 30k steps.
# Settings follow third_party/gsplat/examples/benchmarks/basic.sh so numbers are comparable
# with the reference table in third_party/gsplat/docs/source/tests/eval.rst.
#
# Usage (inside WSL):  bash scripts/m0/run_mipnerf360.sh [scene ...]
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA_DIR=${DATA_DIR:-/root/data/360_v2}
OUT_DIR=${OUT_DIR:-/root/outputs/m0}
SCENES=${*:-"room counter garden"}

cd "$REPO/third_party/gsplat/examples"
for SCENE in $SCENES; do
    case $SCENE in
        bonsai|counter|kitchen|room) FACTOR=2 ;;
        *) FACTOR=4 ;;
    esac
    echo "== $SCENE (data_factor=$FACTOR)"
    python simple_trainer.py default --disable_viewer --data_factor $FACTOR \
        --data_dir "$DATA_DIR/$SCENE/" --result_dir "$OUT_DIR/$SCENE/" \
        2>&1 | tee "$OUT_DIR/$SCENE.log" | grep -E "PSNR|Step:" || true
done

python "$REPO/scripts/m0/collect.py" --out_dir "$OUT_DIR" --scenes $SCENES \
    --save "$REPO/results/m0_mipnerf360.md"
