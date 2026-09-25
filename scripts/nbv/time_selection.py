"""Selection cost per strategy on an idle GPU (the batch runs were parallel, so their timings are not clean).

Usage: python scripts/nbv/time_selection.py --scene /root/data/abo_render/B07J2YB486
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mvr.data import load_scene  # noqa: E402
from mvr.gs import TrainConfig, train  # noqa: E402
from mvr.infogain import gain_matrix, select_greedy  # noqa: E402
from mvr.signals import combined, score_views  # noqa: E402
from mvr.views import initial_views, select_by_score, select_fps, select_random  # noqa: E402


def timed(fn, reps=3):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.time()
        fn()
        torch.cuda.synchronize()
        out.append(time.time() - t)
    return float(np.median(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    args = p.parse_args()
    scene = load_scene(args.scene)
    views = initial_views(scene, "biased", 10, 0)
    splats, t_train = train(scene, views, TrainConfig())
    avail = [i for i in scene.ids("cand") if i not in views]
    rng = np.random.default_rng(0)

    def v1():
        sig = score_views(splats, scene, views, avail)
        select_by_score(scene, avail, combined(sig).cpu().numpy(), 5)

    def ig():
        select_greedy(gain_matrix(splats, scene, views, avail), 5)

    res = {
        "Random": timed(lambda: select_random(avail, 5, rng)),
        "FPS": timed(lambda: select_fps(scene, views, avail, 5)),
        "S1+S2 (v1)": timed(v1),
        "IG (v2)": timed(ig),
    }
    lines = ["# Selection cost (idle RTX 4090 D, chair, 10 training views, 190 candidates, pick 5)", "",
             f"Reference: one 7k-step 3DGS training run = {t_train:.1f} s.", "",
             "| Strategy | Time per selection (median of 3) |", "|---|---|"]
    lines += [f"| {k} | {v * 1000:.1f} ms |" for k, v in res.items()]
    Path("results/m2_selection_cost.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
