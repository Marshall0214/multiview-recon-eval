"""Per-height-band width/depth of the measured furniture cluster (diagnoses measurement-convention gaps).

Usage: python scripts/real/bands.py --data data/real/chair
"""
import sys, json, numpy as np, open3d as o3d
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mvr.real import read_colmap_text, measure_furniture
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/real/chair"); D = ap.parse_args().data
r = json.load(open(f"{D}/result.json"))["alignment"]; s, R, t = r["scale"], np.array(r["R"]), np.array(r["t"])
_, ims = read_colmap_text(f"{D}/work/undist/sparse_txt")
C = np.array([s * R @ (-(im["w2c"][:3,:3].T @ im["w2c"][:3,3])) + t for im in ims])
pts = np.asarray(o3d.io.read_triangle_mesh(f"{D}/work/mesh_metric.ply").sample_points_uniformly(500_000).points)
dims, obj = measure_furniture(pts, C, version="v2")
import cv2
_, _, ang = cv2.minAreaRect(obj[:, :2].astype(np.float32)); a = np.radians(ang)
q = obj[:, :2] @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]).T
# identify which axis is "depth" (shorter overall extent)
ext = np.percentile(q, 99, 0) - np.percentile(q, 1, 0); di = int(np.argmin(ext)); wi = 1 - di
print(f"overall: width {ext[wi]*1000:.0f}  depth {ext[di]*1000:.0f}")
for lo, hi, name in [(0.03, 0.30, "legs 3-30cm"), (0.30, 0.45, "seat 30-45cm"), (0.45, 0.70, "armrest 45-70cm"), (0.70, 0.90, "backrest top 70cm+")]:
    m = (obj[:, 2] >= lo) & (obj[:, 2] < hi)
    if m.sum() < 50: continue
    qq = q[m]
    print(f"{name:20s} n={m.sum():5d}  width {1000*(np.percentile(qq[:,wi],99)-np.percentile(qq[:,wi],1)):.0f}  depth {1000*(np.percentile(qq[:,di],99)-np.percentile(qq[:,di],1)):.0f}  depth-range [{1000*np.percentile(qq[:,di],1):.0f}, {1000*np.percentile(qq[:,di],99):.0f}]")
