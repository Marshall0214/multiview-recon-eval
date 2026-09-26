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
from run_real import fuse, train_gsplat  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--voxel_frac", type=float, default=1 / 1500, help="TSDF voxel / scene extent")
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
    voxel = extent * a.voxel_frac

    mesh = fuse(splats, cams, images, voxel=voxel, max_depth=extent, stride=1)
    pcd = mesh.sample_points_uniformly(int(np.clip(mesh.get_surface_area() / voxel ** 2, 1e5, 2e7)))
    o3d.io.write_point_cloud(str(work / "recon.ply"), pcd)

    with open(work / "recon.log", "w") as f:
        for im in images:
            idx = int(Path(im["name"]).stem) - 1  # T&T image sets: 000001.jpg is frame 0
            c2w = np.linalg.inv(im["w2c"])
            f.write(f"{idx} {idx} 0\n" + "\n".join(" ".join(f"{v:.12f}" for v in row) for row in c2w) + "\n")

    val = sorted((work / "gs" / "stats").glob("val_step*.json"))
    info = {"registered": len(images), "extent_colmap_units": extent, "voxel": voxel,
            "n_points": len(pcd.points), "val": json.loads(val[-1].read_text()) if val else None}
    (work / "recon_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
