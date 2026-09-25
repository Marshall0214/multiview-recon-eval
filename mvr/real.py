"""Real-data helpers (README §4.6): COLMAP text model I/O, ArUco triangulation, metric alignment,
furniture dimension measurement."""

from pathlib import Path

import cv2
import numpy as np

COLS, ROWS, MARKER_MM, GAP_MM = 3, 4, 50.0, 10.0  # must match scripts/real/make_aruco_board.py


# ---------------------------------------------------------------- COLMAP text model

def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_colmap_text(sparse_dir):
    """Undistorted PINHOLE model -> (K per camera id, list of dict(name, cam, w2c))."""
    sparse_dir = Path(sparse_dir)
    cams = {}
    for line in (sparse_dir / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        e = line.split()
        cid, model, w, h = int(e[0]), e[1], int(e[2]), int(e[3])
        p = list(map(float, e[4:]))
        if model == "PINHOLE":
            fx, fy, cx, cy = p
        elif model in ("SIMPLE_PINHOLE",):
            fx = fy = p[0]
            cx, cy = p[1:3]
        else:
            raise ValueError(f"expected undistorted PINHOLE cameras, got {model}")
        cams[cid] = {"K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), "w": w, "h": h}
    images = []
    lines = [ln for ln in (sparse_dir / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    for ln in lines[::2]:  # every other line holds the 2D points
        e = ln.split()
        if len(e) < 10:
            continue
        q, t = np.array(list(map(float, e[1:5]))), np.array(list(map(float, e[5:8])))
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = qvec2rotmat(q), t
        images.append({"name": e[9], "cam": int(e[8]), "w2c": w2c})
    return cams, images


# ---------------------------------------------------------------- ArUco

def board_points(marker_mm=MARKER_MM):
    """{(marker_id, corner_k): xyz in metres} on the board plane (z = 0), scaled to the printed size."""
    s = marker_mm / MARKER_MM
    board = cv2.aruco.GridBoard((COLS, ROWS), MARKER_MM / 1000, GAP_MM / 1000,
                                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    pts = {}
    for mid, obj in zip(board.getIds().flatten(), board.getObjPoints()):
        for k, xyz in enumerate(np.asarray(obj).reshape(4, 3)):
            pts[(int(mid), k)] = xyz * s
    return pts


def detect(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
    corners, ids, _ = det.detectMarkers(img)
    out = {}
    if ids is not None:
        for c, mid in zip(corners, ids.flatten()):
            for k, uv in enumerate(c.reshape(4, 2)):
                out[(int(mid), k)] = uv
    return out


def triangulate(obs, min_views=3):
    """obs: list of (P [3x4], uv). Linear DLT; returns xyz and mean reprojection error (px)."""
    if len(obs) < min_views:
        return None, None
    A = []
    for P, (u, v) in obs:
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    X = np.linalg.svd(np.asarray(A))[2][-1]
    X = X[:3] / X[3]
    err = []
    for P, uv in obs:
        x = P @ np.append(X, 1)
        err.append(np.linalg.norm(x[:2] / x[2] - uv))
    return X, float(np.mean(err))


def umeyama(src, dst):
    """Similarity (s, R, t) minimising |s R src + t - dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


# ---------------------------------------------------------------- measurement

def measure_furniture(points, cam_centers, floor_eps=0.015, cluster_eps=0.03):
    """Metric points in the board frame (z up, floor z=0) -> height / width / depth (mm).

    Keeps points above the floor, clusters them, takes the biggest cluster nearest the orbit centre
    (the object the cameras circle), then measures a robust height and the minimum-area footprint.
    """
    import open3d as o3d

    # 1. keep the volume inside the camera ring: 3DGS also models the far background, which TSDF fuses too
    centre = cam_centers[:, :2].mean(0)
    ring = np.median(np.linalg.norm(cam_centers[:, :2] - centre, axis=1))
    keep = (np.linalg.norm(points[:, :2] - centre, axis=1) < 0.7 * ring) & \
           (points[:, 2] > -0.2) & (points[:, 2] < cam_centers[:, 2].max())
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[keep])).voxel_down_sample(cluster_eps / 3)

    # 2. refine the floor: RANSAC plane on near-floor points (the A4 board only pins the floor locally)
    near = pc.select_by_index(np.where(np.abs(np.asarray(pc.points)[:, 2]) < 0.08)[0])
    (a, b, c, d), _ = near.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=2000)
    n = np.array([a, b, c]) * np.sign(c)
    d = d * np.sign(c)
    pts = np.asarray(pc.points)
    height_above = pts @ n + d                     # signed distance to the fitted floor
    z_axis = n / np.linalg.norm(n)
    x_axis = np.cross([0, 1.0, 0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pts = np.stack([pts @ x_axis, pts @ y_axis, height_above], 1)

    pts = pts[pts[:, 2] > floor_eps]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    labels = np.asarray(pc.cluster_dbscan(eps=cluster_eps, min_points=20))
    centre = np.array([centre @ x_axis[:2], centre @ y_axis[:2]])  # orbit centre in the levelled frame
    best, best_key = None, None
    for lab in set(labels) - {-1}:
        c = pts[labels == lab]
        if len(c) < 200:
            continue
        key = (np.linalg.norm(np.median(c[:, :2], 0) - centre), -len(c))
        if best_key is None or key < best_key:
            best, best_key = c, key
    if best is None:
        raise RuntimeError("no object cluster found above the floor")
    height = np.percentile(best[:, 2], 99.9)  # set on the simulated capture (dev set); thin top edges erode
    (_, _), (w, d), _ = cv2.minAreaRect(best[:, :2].astype(np.float32))
    w, d = max(w, d), min(w, d)
    return {"height": height * 1000, "width": w * 1000, "depth": d * 1000, "n_points": len(best)}, best
