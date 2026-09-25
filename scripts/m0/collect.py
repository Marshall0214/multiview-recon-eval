"""Collect M0 stats and compare against gsplat's published per-scene numbers."""

import argparse
import json
from pathlib import Path

# gsplat-30k / gsplat-7k rows from third_party/gsplat/docs/source/tests/eval.rst (v1.5.3)
REFERENCE = {
    7000: {
        "bicycle": (23.71, 0.668, 0.324), "bonsai": (29.66, 0.922, 0.162),
        "counter": (27.14, 0.878, 0.206), "garden": (26.30, 0.833, 0.123),
        "kitchen": (28.86, 0.902, 0.127), "room": (29.21, 0.893, 0.217),
        "stump": (25.62, 0.720, 0.253),
    },
    30000: {
        "bicycle": (25.22, 0.764, 0.172), "bonsai": (32.06, 0.941, 0.132),
        "counter": (29.02, 0.907, 0.154), "garden": (27.32, 0.865, 0.075),
        "kitchen": (31.16, 0.926, 0.094), "room": (31.36, 0.918, 0.164),
        "stump": (26.53, 0.768, 0.153),
    },
}


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--save", type=Path, required=True)
    args = p.parse_args()

    lines = [
        "# M0: gsplat 3DGS on Mip-NeRF 360",
        "",
        "Reference = gsplat-Xk rows in `third_party/gsplat/docs/source/tests/eval.rst` (TITAN RTX).",
        "Ours = RTX 4090 D, gsplat 1.5.3, `simple_trainer.py default`, test_every=8.",
        "",
        "| Scene | Steps | PSNR (ours / ref) | SSIM (ours / ref) | LPIPS (ours / ref) | #GS | Train time |",
        "|---|---|---|---|---|---|---|",
    ]
    for scene in args.scenes:
        stats_dir = args.out_dir / scene / "stats"
        for step in (7000, 30000):
            val_f = stats_dir / f"val_step{step - 1:04d}.json"
            train_f = stats_dir / f"train_step{step - 1:04d}_rank0.json"
            if not val_f.exists():
                continue
            v = load(val_f)
            t = load(train_f) if train_f.exists() else {}
            ref = REFERENCE[step].get(scene, (float("nan"),) * 3)
            secs = t.get("ellipse_time", float("nan"))
            lines.append(
                f"| {scene} | {step // 1000}k "
                f"| {v['psnr']:.2f} / {ref[0]:.2f} | {v['ssim']:.3f} / {ref[1]:.3f} "
                f"| {v['lpips']:.3f} / {ref[2]:.3f} | {v['num_GS'] / 1e6:.2f}M "
                f"| {secs / 60:.1f} min |"
            )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
