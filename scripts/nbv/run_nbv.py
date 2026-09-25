"""M2: rescan loop (README §4.2-4.4).

For one (scene, init, strategy, seed): R+1 rounds of
    train from scratch on current views -> evaluate on fixed test views
    -> score ALL remaining candidates (S1, S2, true error)  [for Q1, logged for every strategy]
    -> pick N new views with the strategy.

Usage: python scripts/nbv/run_nbv.py --scene /root/data/abo_render/B07J2YB486 --init biased --strategy ours --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mvr.data import load_scene  # noqa: E402
from mvr.geom import load_gt_mesh, raycast_depth  # noqa: E402
from mvr.gs import TrainConfig, render, train  # noqa: E402
from mvr.metrics import evaluate  # noqa: E402
from mvr.signals import combined, score_views  # noqa: E402
from mvr.views import initial_views, select_by_score, select_fps, select_random  # noqa: E402


def gt_depth_cache(scene):
    f = scene.root / "gt_depth_cand.npy"
    ids = scene.ids("cand")
    if not f.exists():
        d = raycast_depth(load_gt_mesh(scene.root / "mesh_gt.ply"), scene.c2w[ids], scene.K,
                          scene.width, scene.height)
        np.save(f, d.astype(np.float16))
    d = np.load(f).astype(np.float32)
    return {i: d[k] for k, i in enumerate(ids)}


@torch.no_grad()
def true_errors(splats, scene, ids, gt_depth, tau_frac=0.01):
    """Per candidate: bad-pixel ratio (geometry) and PSNR (appearance) against GT."""
    white = torch.ones(1, 3, device="cuda")
    bad, psnr = [], []
    for i in ids:
        rgb, alpha, depth, _ = render(splats, scene.viewmats[i:i + 1], scene.K, scene.width, scene.height, bg=white)
        gd = torch.from_numpy(gt_depth[i]).cuda()
        obj = torch.isfinite(gd)
        wrong = (alpha[0] < 0.5) | ((depth[0] - gd).abs() > tau_frac * scene.bbox_diag)
        bad.append(((wrong & obj).sum() / obj.sum().clamp(min=1)).item())
        a = scene.alpha[i][..., None]
        gt = scene.rgb[i] * a + (1 - a)
        mse = ((rgb[0].clamp(0, 1) - gt) ** 2).mean()
        psnr.append((-10 * torch.log10(mse)).item())
    return np.array(bad), np.array(psnr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--init", choices=["uniform", "biased"], required=True)
    p.add_argument("--strategy", choices=["random", "fps", "ours"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--steps", type=int, default=7000)
    p.add_argument("--out", default="/root/outputs/nbv")
    args = p.parse_args()

    scene = load_scene(args.scene)
    gt_pts = np.load(scene.root / "gt_points_observable.npy")
    gt_depth = gt_depth_cache(scene)
    out = Path(args.out) / scene.root.name / args.init / args.strategy / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1000 + args.seed)
    views = initial_views(scene, args.init, args.k, args.seed)  # same init for all strategies
    cfg = TrainConfig(steps=args.steps, refine_stop=int(args.steps * 5 / 7), seed=args.seed)

    for r in range(args.rounds + 1):
        splats, t_train = train(scene, views, cfg)
        res, _ = evaluate(splats, scene, gt_pts)
        res.update({"round": r, "n_views": len(views), "views": [int(v) for v in views], "train_s": t_train})

        if r < args.rounds:
            avail = [i for i in scene.ids("cand") if i not in views]
            t0 = time.time()
            sig = score_views(splats, scene, views, avail)
            score = combined(sig)
            torch.cuda.synchronize()
            res["signal_s"] = time.time() - t0
            bad, psnr = true_errors(splats, scene, avail, gt_depth)
            np.savez(out / f"round{r}_candidates.npz", ids=np.array(avail),
                     S1=sig["S1"].cpu().numpy(), S2=sig["S2"].cpu().numpy(), score=score.cpu().numpy(),
                     bad=bad, psnr=psnr)
            t0 = time.time()
            if args.strategy == "random":
                new = select_random(avail, args.n, rng)
            elif args.strategy == "fps":
                new = select_fps(scene, views, avail, args.n)
            else:
                new = select_by_score(scene, avail, score.cpu().numpy(), args.n)
            res["select_s"] = time.time() - t0 + (res["signal_s"] if args.strategy == "ours" else 0)
            views = views + [int(v) for v in new]

        (out / f"round{r}.json").write_text(json.dumps(res, indent=1))
        print(f"[{scene.root.name}/{args.init}/{args.strategy}/s{args.seed}] round {r}: "
              f"views={res['n_views']} psnr={res['psnr']:.2f} F@1%={res['fscore@1%']:.3f} "
              f"R@1%={res['recall@1%']:.3f} train={t_train:.0f}s", flush=True)


if __name__ == "__main__":
    main()
