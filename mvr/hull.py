"""Visual hull by space carving with training-view masks.

The hull is conservative: space never seen by any training view is kept as "possibly occupied".
It serves (1) as the initialisation volume for 3DGS and (2) as the GT-free reference region
for signal S1 ("hull says maybe object, reconstruction says empty").
"""

import torch


def project(pts, viewmats, K):
    """pts [N,3] -> pixel uv [V,N,2], depth [V,N]."""
    pc = torch.einsum("vij,nj->vni", viewmats[:, :3, :3], pts) + viewmats[:, None, :3, 3]
    z = pc[..., 2]
    uv = torch.einsum("ij,vnj->vni", K, pc)[..., :2] / z.clamp(min=1e-6)[..., None]
    return uv, z


def carve(scene, view_ids, res=128, half=None, mask_thresh=0.5):
    """Return (centers [M,3] of occupied voxels, voxel size)."""
    dev = scene.c2w.device
    half = half or 0.55 * scene.bbox_diag
    lin = torch.linspace(-half, half, res, device=dev)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    occ = torch.ones(len(grid), dtype=torch.bool, device=dev)
    for i in view_ids:
        uv, z = project(grid, scene.viewmats[i:i + 1], scene.K)
        u, v = uv[0, :, 0].round().long(), uv[0, :, 1].round().long()
        inside = (z[0] > 0) & (u >= 0) & (u < scene.width) & (v >= 0) & (v < scene.height)
        m = torch.zeros_like(occ)
        m[inside] = scene.alpha[i][v[inside], u[inside]] < mask_thresh
        occ &= ~m  # carve voxels that project onto background
    return grid[occ], (2 * half) / (res - 1)


@torch.no_grad()
def silhouette(centers, voxel, viewmats, K, width, height):
    """Hull silhouettes [C,H,W] (bool): rasterise voxels as opaque isotropic Gaussians with gsplat."""
    from gsplat import rasterization

    n = len(centers)
    dev = centers.device
    _, alpha, _ = rasterization(
        means=centers, quats=torch.tensor([[1.0, 0, 0, 0]], device=dev).expand(n, 4),
        scales=torch.full((n, 3), 0.5 * voxel, device=dev), opacities=torch.full((n,), 0.99, device=dev),
        colors=torch.zeros((n, 1), device=dev), viewmats=viewmats, Ks=K[None].expand(len(viewmats), 3, 3),
        width=width, height=height, packed=False)
    return alpha[..., 0] > 0.5
