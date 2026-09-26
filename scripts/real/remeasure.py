"""Re-measure a finished run (DATA/work/mesh_metric.ply) with measurement v1 and v2, no re-training.

Usage: python scripts/real/remeasure.py --data data/real/chair
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mvr.real import measure_furniture, read_colmap_text  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
a = p.parse_args()
res = json.loads((a.data / "result.json").read_text())
meas = json.loads((a.data / "measurements.json").read_text())
s, R, t = res["alignment"]["scale"], np.array(res["alignment"]["R"]), np.array(res["alignment"]["t"])
_, images = read_colmap_text(a.data / "work" / "undist" / "sparse_txt")
centres = np.array([s * R @ (-(im["w2c"][:3, :3].T @ im["w2c"][:3, 3])) + t for im in images])
pts = np.asarray(o3d.io.read_triangle_mesh(str(a.data / "work" / "mesh_metric.ply")).sample_points_uniformly(500_000).points)
out = {}
for v in ("v1", "v2"):
    dims, _ = measure_furniture(pts, centres, version=v)
    out[v] = {k: {"tape_mm": meas["dims_mm"][k], "recon_mm": round(dims[k], 1),
                  "err_mm": round(dims[k] - meas["dims_mm"][k], 1),
                  "err_pct": round(100 * (dims[k] - meas["dims_mm"][k]) / meas["dims_mm"][k], 2)}
              for k in ("height", "width", "depth")}
res["measurement_versions"] = out
(a.data / "result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(out, indent=1))
