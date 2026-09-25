"""M4 real-data pipeline: photos -> COLMAP -> ArUco metric alignment -> 3DGS -> mesh -> furniture dims.

Usage: python scripts/real/run_real.py --data data/real/chair [--steps 15000]
  expects DATA/images/*.jpg and DATA/measurements.json (see docs/capture_guide.md)
Writes DATA/work/ (COLMAP, gsplat) and DATA/result.json.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mvr.geom import tsdf_fuse  # noqa: E402
from mvr.gs import render  # noqa: E402
from mvr.real import board_points, detect, measure_furniture, read_colmap_text, triangulate, umeyama  # noqa: E402


def sh(*cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, stdout=subprocess.DEVNULL)


def run_colmap(images, work, max_size):
    db, sparse, undist = work / "db.db", work / "sparse", work / "undist"
    if (undist / "sparse" / "0").exists():
        return undist
    sparse.mkdir(parents=True, exist_ok=True)
    sh("colmap", "feature_extractor", "--database_path", db, "--image_path", images,
       "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "OPENCV",
       "--SiftExtraction.use_gpu", 0, "--SiftExtraction.max_image_size", max_size)
    sh("colmap", "exhaustive_matcher", "--database_path", db, "--SiftMatching.use_gpu", 0)
    sh("colmap", "mapper", "--database_path", db, "--image_path", images, "--output_path", sparse)
    models = sorted(sparse.iterdir(), key=lambda p: -sum(1 for _ in p.iterdir()))
    sh("colmap", "image_undistorter", "--image_path", images, "--input_path", models[0],
       "--output_path", undist, "--output_type", "COLMAP", "--max_image_size", max_size)
    (undist / "sparse" / "0").mkdir(parents=True)
    for f in (undist / "sparse").glob("*.bin"):
        shutil.move(str(f), undist / "sparse" / "0" / f.name)
    txt = undist / "sparse_txt"
    txt.mkdir(exist_ok=True)
    sh("colmap", "model_converter", "--input_path", undist / "sparse" / "0", "--output_path", txt,
       "--output_type", "TXT")
    return undist


def align_to_board(undist, marker_mm, max_reproj=2.0):
    """Similarity COLMAP -> board frame (metres, floor z=0, cameras at z>0)."""
    cams, images = read_colmap_text(undist / "sparse_txt")
    obs = {}
    for im in images:
        P = cams[im["cam"]]["K"] @ im["w2c"][:3]
        for key, uv in detect(undist / "images" / im["name"]).items():
            obs.setdefault(key, []).append((P, uv))
    ref = board_points(marker_mm)
    src, dst = [], []
    for key, o in obs.items():
        X, err = triangulate(o)
        if X is not None and err < max_reproj and key in ref:
            src.append(X)
            dst.append(ref[key])
    if len(src) < 8:
        raise RuntimeError(f"only {len(src)} ArUco corners triangulated; is the board visible in >=3 photos?")
    src, dst = np.array(src), np.array(dst)
    s, R, t = umeyama(src, dst)
    centres = np.array([-(im["w2c"][:3, :3].T @ im["w2c"][:3, 3]) for im in images])
    if np.mean((s * centres @ R.T + t)[:, 2]) < 0:  # board z axis points into the floor: flip
        F = np.diag([1.0, -1.0, -1.0])
        R, t, dst = F @ R, F @ t, dst @ F.T
    rms = float(np.sqrt(((s * src @ R.T + t - dst) ** 2).sum(1).mean()))
    return {"scale": float(s), "R": R.tolist(), "t": t.tolist(), "corners": len(src), "rms_mm": rms * 1000,
            "n_images": len(images)}, cams, images


def train_gsplat(undist, out, steps):
    ckpt = out / "ckpts" / f"ckpt_{steps - 1}_rank0.pt"
    if not ckpt.exists():
        sh(sys.executable, REPO / "third_party/gsplat/examples/simple_trainer.py", "default",
           "--disable_viewer", "--data_dir", undist, "--data_factor", 1, "--result_dir", out,
           "--no-normalize_world_space", "--max_steps", steps, "--eval_steps", steps,
           "--save_steps", steps, "--ply_steps", steps)
    return torch.load(ckpt, map_location="cuda")["splats"]


@torch.no_grad()
def fuse(splats, cams, images, voxel, max_depth, stride=2):
    splats = {k: v.cuda() for k, v in splats.items()}
    depths, rgbs, c2ws = [], [], []
    K0 = None
    for im in images[::stride]:
        cam = cams[im["cam"]]
        K = torch.tensor(cam["K"], dtype=torch.float32, device="cuda")
        vm = torch.tensor(im["w2c"], dtype=torch.float32, device="cuda")[None]
        rgb, alpha, depth, _ = render(splats, vm, K, cam["w"], cam["h"], bg=torch.ones(1, 3, device="cuda"))
        d = depth[0].clone()
        d[alpha[0] < 0.5] = 0
        depths.append(d.cpu().numpy())
        rgbs.append(rgb[0].clamp(0, 1).cpu().numpy())
        c2ws.append(np.linalg.inv(im["w2c"]))
        K0 = cam
    return tsdf_fuse(depths, rgbs, np.array(c2ws), K0["K"], K0["w"], K0["h"], voxel, max_depth=max_depth)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--max_size", type=int, default=1600)
    p.add_argument("--voxel_mm", type=float, default=4.0)
    args = p.parse_args()

    meas = json.loads((args.data / "measurements.json").read_text())
    work = args.data / "work"
    undist = run_colmap(args.data / "images", work, args.max_size)
    sim3, cams, images = align_to_board(undist, meas["marker_mm"])
    print("alignment:", json.dumps(sim3))
    splats = train_gsplat(undist, work / "gs", args.steps)

    s, R, t = sim3["scale"], np.array(sim3["R"]), np.array(sim3["t"])
    centres = np.array([s * R @ (-(im["w2c"][:3, :3].T @ im["w2c"][:3, 3])) + t for im in images])
    ring = np.median(np.linalg.norm(centres[:, :2] - centres[:, :2].mean(0), axis=1))  # metres
    # fuse only up to the far side of the object: 3DGS also models the whole floor and background
    mesh = fuse(splats, cams, images, voxel=args.voxel_mm / 1000 / s, max_depth=1.7 * ring / s)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, t
    mesh.transform(T)
    v = np.asarray(mesh.vertices)
    outside = np.linalg.norm(v[:, :2] - centres[:, :2].mean(0), axis=1) > 0.7 * ring
    mesh.remove_vertices_by_mask(outside)
    o3d.io.write_triangle_mesh(str(work / "mesh_metric.ply"), mesh)
    pts = np.asarray(mesh.sample_points_uniformly(500_000).points)
    dims, _ = measure_furniture(pts, centres)

    rows = {}
    for k in ("height", "width", "depth"):
        if k in meas.get("dims_mm", {}):
            gt = meas["dims_mm"][k]
            rows[k] = {"tape_mm": gt, "recon_mm": dims[k], "err_mm": dims[k] - gt, "err_pct": 100 * (dims[k] - gt) / gt}
    result = {"alignment": sim3, "dims": rows, "n_object_points": dims["n_points"]}
    (args.data / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
