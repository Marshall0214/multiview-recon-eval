"""Train 3DGS on an initial view set and evaluate (image + geometry metrics, signal timing).

Usage: python scripts/eval/train_eval_once.py --scene /root/data/abo_render/B07J2YB486 --init uniform --steps 7000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mvr.data import load_scene  # noqa: E402
from mvr.gs import TrainConfig, train  # noqa: E402
from mvr.metrics import evaluate  # noqa: E402
from mvr.views import initial_views  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--init", default="uniform", choices=["uniform", "biased", "all"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--steps", type=int, default=7000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    scene = load_scene(args.scene)
    gt = np.load(Path(args.scene) / "gt_points_observable.npy")
    ids = scene.ids("cand") if args.init == "all" else initial_views(scene, args.init, args.k, args.seed)
    cfg = TrainConfig(steps=args.steps, refine_stop=int(args.steps * 5 / 7), seed=args.seed)
    splats, t_train = train(scene, ids, cfg, log_every=1000)
    t0 = time.time()
    res, _ = evaluate(splats, scene, gt)
    res.update({"scene": Path(args.scene).name, "init": args.init, "n_views": len(ids),
                "steps": args.steps, "train_s": t_train, "eval_s": time.time() - t0})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
