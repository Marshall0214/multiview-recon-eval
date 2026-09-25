"""M1 self-check of the evaluation pipeline (README §4.5).

1. Camera convention: GT ray-cast hit mask vs Blender alpha mask -> IoU should be ~1.
2. Geometry pipeline: TSDF-fuse GT depth from all candidate views -> F-score vs GT mesh should be ~1.

Usage: python scripts/eval/check_gt_pipeline.py --scene /root/data/abo_render/B07J2YB486
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mvr.data import load_scene  # noqa: E402
from mvr.geom import eval_mesh, load_gt_mesh, observable_gt_points, raycast_depth, tsdf_fuse  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--voxel_frac", type=float, default=0.002, help="TSDF voxel as fraction of bbox diag")
    args = p.parse_args()

    scene = load_scene(args.scene)
    mesh = load_gt_mesh(Path(args.scene) / "mesh_gt.ply")
    ids = scene.ids("cand")
    depth = raycast_depth(mesh, scene.c2w[ids], scene.K, scene.width, scene.height)

    hit = np.isfinite(depth)
    fg = scene.alpha[ids].cpu().numpy() > 0.5
    iou = (hit & fg).sum() / (hit | fg).sum()

    fused = tsdf_fuse(depth, scene.rgb[ids].cpu().numpy(), scene.c2w[ids].cpu().numpy(),
                      scene.K.cpu().numpy(), scene.width, scene.height,
                      voxel=args.voxel_frac * scene.bbox_diag)
    all_ids = list(range(len(scene.names)))
    obs = observable_gt_points(mesh, scene.c2w[all_ids], scene.K, scene.width, scene.height, scene.bbox_diag)
    np.save(Path(args.scene) / "gt_points_observable.npy", obs)
    res = {
        "scene": Path(args.scene).name,
        "mask_iou": float(iou),
        "observable_frac": len(obs) / 200_000,
        "vs_all_surface": eval_mesh(fused, mesh, scene.bbox_diag),
        "vs_observable": eval_mesh(fused, obs, scene.bbox_diag),
    }
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
