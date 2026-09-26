"""3DGS reconstruction of a Tanks-and-Temples scene for evaluation (real-data Q3).

COLMAP (scripts/tnt/run_colmap_tnt.py) -> gsplat simple_trainer (held-out every 8th frame, PSNR logged)
-> render expected depth at all registered views -> TSDF -> point cloud + trajectory (.log) for eval_tnt.py.

Usage: python scripts/tnt/recon_tnt.py --scene_dir /root/data/tnt/Meetingroom [--steps 30000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "real"))
sys.path.insert(0, str(REPO))
from mvr.real import read_colmap_text  # noqa: E402
import torch  # noqa: E402
from mvr.gs import render  # noqa: E402
from run_real import train_gsplat  # noqa: E402


@torch.no_grad()
def backproject(splats, cams, images, max_depth, voxel, pixel_stride=2, chunk=20, alpha_min=0.5):
    """Point cloud from rendered expected depth of every view (like MVS depth-map fusion).

    TSDF fusion was tried first: with 3DGS depth noise on a 371-frame full-HD room it allocates ~0.5 GB of
    voxel blocks per frame (OOM at 64 GB). Back-projection + periodic voxel downsampling keeps memory flat.
    """
    splats = {k: v.cuda() for k, v in splats.items()}
    acc, parts = o3d.geometry.PointCloud(), []
    for n, im in enumerate(images):
        cam = cams[im["cam"]]
        K = torch.tensor(cam["K"], dtype=torch.float32, device="cuda")
        vm = torch.tensor(im["w2c"], dtype=torch.float32, device="cuda")
        _, alpha, depth, _ = render(splats, vm[None], K, cam["w"], cam["h"])
        d, a = depth[0][::pixel_stride, ::pixel_stride], alpha[0][::pixel_stride, ::pixel_stride]
        v, u = torch.meshgrid(torch.arange(0, cam["h"], pixel_stride, device="cuda"),
                              torch.arange(0, cam["w"], pixel_stride, device="cuda"), indexing="ij")
        ok = (a > alpha_min) & (d > 0) & (d < max_depth)
        z = d[ok]
        x = (u[ok] - K[0, 2]) / K[0, 0] * z
        y = (v[ok] - K[1, 2]) / K[1, 1] * z
        pc = torch.stack([x, y, z], 1)
        c2w = torch.linalg.inv(vm)
        parts.append((pc @ c2w[:3, :3].T + c2w[:3, 3]).cpu().numpy())
        if (n + 1) % chunk == 0 or n == len(images) - 1:
            acc.points.extend(o3d.utility.Vector3dVector(np.concatenate(parts)))
            acc = acc.voxel_down_sample(voxel)
            parts = []
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--voxel_frac", type=float, default=1 / 1500, help="TSDF voxel / scene extent (if no metric scale)")
    # depth_trunc 4.5 m as in the 2DGS release scripts for "large" T&T scenes (incl. Meetingroom),
    # adopted as-is rather than tuned against the GT. They are in metres, so they need the COLMAP->metric
    # scale; we take it from aligning our trajectory to the benchmark's reference COLMAP trajectory.
    p.add_argument("--metric_scale", type=float, default=None, help="metres per COLMAP unit")
    p.add_argument("--voxel_m", type=float, default=0.003, help="point-cloud voxel; eval downsamples to tau/2 = 5 mm")
    p.add_argument("--depth_trunc_m", type=float, default=4.5)
    a = p.parse_args()
    work = a.scene_dir / "work"
    undist = work / "undist"

    splats = train_gsplat(undist, work / "gs", a.steps)
    cams, images = read_colmap_text(undist / "sparse_txt")
    images = sorted(images, key=lambda im: im["name"])

    # scene extent in COLMAP units from the sparse points (robust)
    pts = []
    for ln in (undist / "sparse_txt" / "points3D.txt").read_text().splitlines():
        if ln and not ln.startswith("#"):
            pts.append(list(map(float, ln.split()[1:4])))
    pts = np.array(pts)
    lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
    extent = float(np.linalg.norm(hi - lo))
    if a.metric_scale:
        voxel, max_depth = a.voxel_m / a.metric_scale, a.depth_trunc_m / a.metric_scale
    else:
        voxel, max_depth = extent * a.voxel_frac, extent

    pcd = backproject(splats, cams, images, max_depth=max_depth, voxel=voxel)
    o3d.io.write_point_cloud(str(work / "recon.ply"), pcd)

    with open(work / "recon.log", "w") as f:
        for im in images:
            idx = int(Path(im["name"]).stem) - 1  # T&T image sets: 000001.jpg is frame 0
            c2w = np.linalg.inv(im["w2c"])
            f.write(f"{idx} {idx} 0\n" + "\n".join(" ".join(f"{v:.12f}" for v in row) for row in c2w) + "\n")

    val = sorted((work / "gs" / "stats").glob("val_step*.json"))
    info = {"registered": len(images), "extent_colmap_units": extent, "voxel": voxel, "max_depth": max_depth,
            "metric_scale": a.metric_scale,
            "n_points": len(pcd.points), "val": json.loads(val[-1].read_text()) if val else None}
    (work / "recon_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
