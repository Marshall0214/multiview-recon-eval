"""GT-free quality signals for candidate views (README §3).

S1 hull-hole:   fraction of pixels inside the visual-hull silhouette whose rendered alpha < tau.
                "The hull says there may be object here, the reconstruction says empty / unsure."
S2 low-support: alpha-weighted mean of 1/(1+c_g) over the rendered pixels, where c_g is the number of
                training views in which Gaussian g actually contributes to the image (occlusion-aware:
                measured by d(sum alpha)/d(opacity) > eps, so fully occluded Gaussians do not count).
"""

import torch

from .gs import render
from .hull import carve, silhouette


def support_counts(splats, scene, train_ids, eps=1e-6):
    """c_g for every Gaussian: in how many training views it has non-zero alpha contribution."""
    counts = torch.zeros(len(splats["means"]), device="cuda")
    op = splats["opacities"]
    for i in train_ids:
        op.grad = None
        _, alpha, _, _ = render(splats, scene.viewmats[i:i + 1], scene.K, scene.width, scene.height, mode="RGB")
        alpha.sum().backward()
        counts += (op.grad.abs() > eps).float()
    op.grad = None
    return counts


@torch.no_grad()
def score_views(splats, scene, train_ids, cand_ids, alpha_tau=0.5, hull_res=128):
    """Return dict of per-candidate signals (tensors aligned with cand_ids)."""
    with torch.enable_grad():
        counts = support_counts(splats, scene, train_ids)
    inv_support = (1.0 / (1.0 + counts))[:, None]  # [N,1] as a 1-channel "colour"
    centers, voxel = carve(scene, train_ids, res=hull_res)

    s1, s2 = [], []
    for i in cand_ids:
        vm = scene.viewmats[i:i + 1]
        sup, alpha, _, _ = render(splats, vm, scene.K, scene.width, scene.height,
                                  colors=inv_support, mode="RGB")
        alpha = alpha[0]
        hull = silhouette(centers, voxel, vm, scene.K, scene.width, scene.height)[0]
        s1.append(((alpha < alpha_tau) & hull).float().sum() / hull.float().sum().clamp(min=1))
        # sup is alpha-premultiplied: normalise by total alpha
        s2.append(sup[0, ..., 0].sum() / alpha.sum().clamp(min=1e-6))
    return {"S1": torch.stack(s1), "S2": torch.stack(s2)}


def zscore(x):
    return (x - x.mean()) / x.std().clamp(min=1e-8)


def combined(sig):
    """Equal-weight z-score sum (weights fixed a priori, README §3)."""
    return zscore(sig["S1"]) + zscore(sig["S2"])
