"""Tanks-and-Temples F-score, ported from the official python_toolbox (isl-org/TanksAndTemples,
evaluation/run.py + registration.py + evaluation.py) to the Open3D >= 0.10 API.

Same procedure and parameters as the official script:
  1. rough alignment: RANSAC similarity between our camera centres and the reference COLMAP trajectory
     (then the provided *_trans.txt to the GT frame);
  2. refinement: voxel ICP (tau, 80 tau) -> voxel ICP (tau/2, 20 tau) -> uniform ICP (2 tau), with scale;
  3. crop with the provided polygon volume, voxel-downsample both clouds at tau/2, P/R/F at tau.
One deliberate difference: correspondences in step 1 are matched by frame index, because our COLMAP
run may not register every frame (the official code assumes identical, complete trajectories).

Usage: python scripts/tnt/eval_tnt.py --gt_dir DIR/Meetingroom --ply recon.ply --log recon.log
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d

TAU = {"Barn": 0.01, "Caterpillar": 0.005, "Church": 0.025, "Courthouse": 0.025,
       "Ignatius": 0.003, "Meetingroom": 0.01, "Truck": 0.005}  # official scenes_tau_dict
MAX_POINT_NUMBER = 4e6
reg = o3d.pipelines.registration


def read_log(path):
    """{frame_index: 4x4 camera-to-world}. Frame index = first metadata field."""
    lines = Path(path).read_text().split("\n")
    out = {}
    for k in range(0, len(lines) - 4, 5):
        if not lines[k].strip():
            continue
        idx = int(lines[k].split()[0])
        out[idx] = np.array([list(map(float, lines[k + j].split())) for j in range(1, 5)])
    return out


def crop_down(pcd, vol, method, voxel=0.01, trans=np.eye(4)):
    p = copy.deepcopy(pcd)
    p.transform(trans)
    p = vol.crop_point_cloud(p)
    if method == "voxel":
        return p.voxel_down_sample(voxel)
    n = len(p.points)
    if n > MAX_POINT_NUMBER:
        return p.uniform_down_sample(int(round(n / MAX_POINT_NUMBER)))
    return p


def icp(src, tgt, init, vol, thr, max_itr, voxel=None, with_scale=True):
    method = "voxel" if voxel else "uniform"
    s = crop_down(src, vol, method, voxel, init)
    t = crop_down(tgt, vol, method, voxel)
    r = reg.registration_icp(s, t, thr, np.eye(4), reg.TransformationEstimationPointToPoint(with_scale),
                             reg.ICPConvergenceCriteria(1e-6, max_itr))
    return r.transformation @ init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", type=Path, required=True)
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--pre_voxel", action="store_true",
                   help="voxel-downsample both clouds to tau/2 at load. The final P/R/F step downsamples to tau/2 "
                        "anyway, so the metric is unchanged; only the ICP inputs get sparser. Needed for noisy 3DGS "
                        "clouds (91M points on Meetingroom: the unmodified pipeline ran > 1.5 h without finishing).")
    p.add_argument("--rigid_icp", action="store_true",
                   help="keep the trajectory-alignment scale; refine rotation/translation only. The official ICP "
                        "also optimises scale, which floaters in a 3DGS cloud can drag (3.5%% on Meetingroom).")
    a = p.parse_args()
    scene = a.gt_dir.name
    tau = TAU[scene]

    pcd = o3d.io.read_point_cloud(str(a.ply))
    gt = o3d.io.read_point_cloud(str(a.gt_dir / f"{scene}.ply"))
    if a.pre_voxel:
        pcd, gt = pcd.voxel_down_sample(tau / 2), gt.voxel_down_sample(tau / 2)
        print(f"pre-voxel {tau / 2} m: recon {len(pcd.points)} pts, gt {len(gt.points)} pts", flush=True)
    gt_trans = np.loadtxt(a.gt_dir / f"{scene}_trans.txt")
    vol = o3d.visualization.read_selection_polygon_volume(str(a.gt_dir / f"{scene}.json"))

    ours, ref = read_log(a.log), read_log(a.gt_dir / f"{scene}_COLMAP_SfM.log")
    common = sorted(set(ours) & set(ref))
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([ours[i][:3, 3] for i in common])))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([ref[i][:3, 3] for i in common])))
    tgt.transform(gt_trans)
    corres = o3d.utility.Vector2iVector(np.array([[k, k] for k in range(len(common))]))
    rough = reg.registration_ransac_based_on_correspondence(
        src, tgt, corres, 0.2, reg.TransformationEstimationPointToPoint(True), 6, [],
        reg.RANSACConvergenceCriteria(100000, 0.999)).transformation

    ws = not a.rigid_icp
    T = icp(pcd, gt, rough, vol, tau * 80, 20, voxel=tau, with_scale=ws)
    T = icp(pcd, gt, T, vol, tau * 20, 20, voxel=tau / 2, with_scale=ws)
    T = icp(pcd, gt, T, vol, 2 * tau, 20, with_scale=ws)

    s = copy.deepcopy(pcd)
    s.transform(T)
    s = vol.crop_point_cloud(s).voxel_down_sample(tau / 2)
    t = vol.crop_point_cloud(copy.deepcopy(gt)).voxel_down_sample(tau / 2)
    d1 = np.asarray(s.compute_point_cloud_distance(t))
    d2 = np.asarray(t.compute_point_cloud_distance(s))
    prec, rec = float((d1 < tau).mean()), float((d2 < tau).mean())
    rough_scale = float(np.cbrt(np.linalg.det(rough[:3, :3])))
    res = {"scene": scene, "tau_m": tau, "icp": "rigid" if a.rigid_icp else "similarity (official)",
           "scale_from_trajectory": rough_scale, "frames_matched": len(common), "frames_reference": len(ref),
           "precision": prec, "recall": rec, "fscore": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
           "scale_to_gt": float(np.cbrt(np.linalg.det(T[:3, :3]))),
           "median_recon_to_gt_m": float(np.median(d1)), "median_gt_to_recon_m": float(np.median(d2)),
           "n_recon_pts_cropped": len(d1), "n_gt_pts_cropped": len(d2),
           "rough_vs_final_shift_m": float(np.linalg.norm((T - rough)[:3, 3])),
           "transform": T.tolist()}
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
