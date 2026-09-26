"""Detailed chair dimensions on the reconstruction, using the same definitions as the tape protocol
(docs/capture_guide.md). Extents are 1st-99th percentiles, as in measurement v2; nothing here is tuned to
the tape values.

Chair frame: z up from the board-plane floor; the depth axis is the footprint axis along which the
backrest top is offset from the legs, +y = back; x = width.

Usage: python scripts/real/measure_detail.py --data data/real/chair
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mvr.real import measure_furniture, read_colmap_text  # noqa: E402

LO, HI = 1, 99


def ext(v):
    return float(np.percentile(v, HI) - np.percentile(v, LO)) if len(v) > 20 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/real/chair"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = json.loads((a.data / "result.json").read_text())
    tape = json.loads((a.data / "measurements.json").read_text())["dims_mm"]
    s, R, t = res["alignment"]["scale"], np.array(res["alignment"]["R"]), np.array(res["alignment"]["t"])
    _, ims = read_colmap_text(a.data / "work" / "undist" / "sparse_txt")
    C = np.array([s * R @ (-(im["w2c"][:3, :3].T @ im["w2c"][:3, 3])) + t for im in ims])
    o3d.utility.random.seed(a.seed)
    pts = np.asarray(o3d.io.read_triangle_mesh(str(a.data / "work" / "mesh_metric.ply"))
                     .sample_points_uniformly(500_000).points)
    dims, obj = measure_furniture(pts, C, version="v2")

    # chair frame
    _, _, ang = cv2.minAreaRect(obj[:, :2].astype(np.float32))
    r = np.radians(ang)
    axes = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
    q = obj[:, :2] @ axes.T
    z = obj[:, 2]
    top, legs = z > 0.70, z < 0.10

    def mid(v):
        return 0.5 * (np.percentile(v, LO) + np.percentile(v, HI))

    # the backrest is a thin plate: its extent is small along depth and large along width. (Medians of the
    # few, unevenly sampled leg points are not reliable enough to find the axis from the backrest offset.)
    di = int(np.argmin([ext(q[top, 0]), ext(q[top, 1])]))
    wi = 1 - di
    y = q[:, di] * np.sign(mid(q[top, di]) - mid(q[legs, di]))  # +y = back
    x = q[:, wi]
    front = y < mid(y[legs])
    cx = mid(x[legs])

    out = {}
    out["height"] = float(np.percentile(z, 99.9))
    seat_zone = front & (np.abs(x - cx) < 0.15) & (z > 0.25) & (z < 0.55)
    out["seat_height"] = float(np.percentile(z[seat_zone], 95))
    arm_zone = front & (np.abs(x - cx) > 0.20) & (z > 0.45) & (z < 0.80)
    out["armrest_height"] = float(np.percentile(z[arm_zone], 99))
    arm_band = (z > out["armrest_height"] - 0.10) & (z < out["armrest_height"] + 0.02) & front
    out["width_armrest"] = ext(x[arm_band])
    out["width_front_legs"] = ext(x[legs & front])
    seat_band = (z > out["seat_height"] - 0.05) & (z < out["seat_height"] + 0.01)
    out["width_seat"] = ext(x[seat_band & (np.abs(x - cx) < 0.35)])
    out["depth_seat"] = ext(y[seat_band])
    out["depth_legs"] = ext(y[legs])
    out["overall_depth"] = ext(y)
    out["backrest_overhang"] = float(np.percentile(y, HI) - np.percentile(y[legs], HI))

    lines = ["| Dimension | Tape (mm) | Reconstruction (mm) | Error (mm) | Error (%) |", "|---|---|---|---|---|"]
    rows = {}
    for k, v in out.items():
        v *= 1000
        if k in tape and tape[k]:
            e = v - tape[k]
            rows[k] = {"tape_mm": tape[k], "recon_mm": round(v, 1), "err_mm": round(e, 1),
                       "err_pct": round(100 * e / tape[k], 1)}
            lines.append(f"| {k} | {tape[k]} | {v:.0f} | {e:+.0f} | {100 * e / tape[k]:+.1f} |")
    errs = np.array([abs(r["err_mm"]) for r in rows.values()])
    lines.append(f"\nMean |error| {errs.mean():.1f} mm, median {np.median(errs):.1f} mm, max {errs.max():.1f} mm "
                 f"over {len(errs)} dimensions.")
    print("\n".join(lines))
    res["detailed_dims"] = rows
    (a.data / "result.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
