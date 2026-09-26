"""Geometry evaluation: GT depth by ray casting, TSDF fusion, Chamfer-style F-score (README §4.4)."""

import numpy as np
import open3d as o3d
import torch


def load_gt_mesh(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    return mesh


def raycast_depth(mesh, c2w, K, width, height):
    """GT z-depth maps [V,H,W] (inf where no hit) for OpenCV c2w matrices."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    Knp = K.cpu().numpy().astype(np.float64)
    out = []
    for M in c2w.cpu().numpy():
        w2c = np.linalg.inv(M)
        rays = scene.create_rays_pinhole(Knp, w2c, width, height)
        t = scene.cast_rays(rays)["t_hit"].numpy()  # distance along (unnormalised) ray dir
        d = rays.numpy()[..., 3:]
        # z-depth = t * (dir . camera_z) ; camera z axis in world = M[:3, 2]
        z = t * (d @ M[:3, 2])
        out.append(z)
    return np.stack(out)


def tsdf_fuse(depths, rgbs, c2w, K, width, height, voxel, trunc_mult=4.0, max_depth=100.0):
    """Fuse z-depth maps [V,H,W] (0 or inf = invalid) into a triangle mesh."""
    return tsdf_fuse_stream(zip(depths, rgbs, c2w), K, width, height, voxel, trunc_mult, max_depth)


def tsdf_fuse_stream(frames, K, width, height, voxel, trunc_mult=4.0, max_depth=100.0):
    """Like tsdf_fuse, but `frames` yields (depth, rgb, c2w) one at a time, so memory does not grow with
    the number of views (371 full-HD frames of a T&T scene do not fit in RAM at once)."""
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc_mult * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, float(K[0, 0]), float(K[1, 1]),
                                             float(K[0, 2]), float(K[1, 2]))
    for d, c, M in frames:
        d = np.where(np.isfinite(d), d, 0).astype(np.float32)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray((c * 255).astype(np.uint8))),
            o3d.geometry.Image(d), depth_scale=1.0, depth_trunc=max_depth,
            convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(M))
    return vol.extract_triangle_mesh()


def sample_points(mesh, n=200_000, seed=0):
    o3d.utility.random.seed(seed)
    return np.asarray(mesh.sample_points_uniformly(n).points)


def fscore(pred_pts, gt_pts, tau):
    """Precision / recall / F at distance tau (Tanks-and-Temples style)."""
    if len(pred_pts) == 0:
        return {"precision": 0.0, "recall": 0.0, "fscore": 0.0}
    pp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_pts))
    gp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts))
    d_p2g = np.asarray(pp.compute_point_cloud_distance(gp))
    d_g2p = np.asarray(gp.compute_point_cloud_distance(pp))
    p = float((d_p2g < tau).mean())
    r = float((d_g2p < tau).mean())
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "fscore": f,
            "chamfer": float(d_p2g.mean() + d_g2p.mean()) / 2}


def observable_gt_points(gt_mesh, c2w, K, width, height, bbox_diag, n=200_000, rel_eps=0.002):
    """GT surface samples visible from at least one of the given views.

    Uniform samples of the whole mesh include surfaces no camera can ever see (the underside against the
    floor, internal faces), which caps recall below 1 no matter how good the scan is. Like DTU's
    observation masks, recall is measured only against points some view in the full pool can observe.
    """
    pts = sample_points(gt_mesh, n)
    depth = raycast_depth(gt_mesh, c2w, K, width, height)
    Knp = K.cpu().numpy() if hasattr(K, "cpu") else K
    c2w = c2w.cpu().numpy() if hasattr(c2w, "cpu") else c2w
    seen = np.zeros(len(pts), bool)
    for M, D in zip(c2w, depth):
        pc = (pts - M[:3, 3]) @ M[:3, :3]  # world -> camera (R^T (p - t))
        z = pc[:, 2]
        uv = pc @ Knp.T
        u = np.round(uv[:, 0] / np.maximum(z, 1e-9)).astype(int)
        v = np.round(uv[:, 1] / np.maximum(z, 1e-9)).astype(int)
        ok = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        d = np.full(len(pts), np.inf)
        d[ok] = D[v[ok], u[ok]]
        seen |= ok & (np.abs(d - z) < rel_eps * bbox_diag)
    return pts[seen]


def eval_mesh(pred_mesh, gt, bbox_diag, taus=(0.005, 0.01, 0.02), n=200_000):
    """gt: a TriangleMesh (sampled uniformly) or an [M,3] array of (observable) GT points."""
    pred = sample_points(pred_mesh, n) if len(pred_mesh.triangles) else np.zeros((0, 3))
    if not isinstance(gt, np.ndarray):
        gt = sample_points(gt, n)
    out = {}
    for t in taus:
        r = fscore(pred, gt, t * bbox_diag)
        out.update({f"{k}@{t * 100:g}%": v for k, v in r.items() if k != "chamfer"})
        out["chamfer"] = r.get("chamfer", float("nan"))
    return out


@torch.no_grad()
def fuse_from_splats(splats, scene, view_ids, render_fn, voxel, alpha_min=0.5):
    """Render depth/rgb from the reconstruction at `view_ids`, fuse into a mesh."""
    depths, rgbs = [], []
    for i in view_ids:
        rgb, alpha, depth, _ = render_fn(splats, scene.viewmats[i:i + 1], scene.K, scene.width, scene.height,
                                         bg=torch.ones(1, 3, device="cuda"))
        d = depth[0].clone()  # "ED" is already normalised by alpha in gsplat
        d[alpha[0] < alpha_min] = 0
        depths.append(d.cpu().numpy())
        rgbs.append(rgb[0].clamp(0, 1).cpu().numpy())
    return tsdf_fuse(depths, rgbs, scene.c2w[view_ids].cpu().numpy(), scene.K.cpu().numpy(),
                     scene.width, scene.height, voxel)
