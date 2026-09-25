"""Evaluate a reconstruction on the fixed test views (README §4.4)."""

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .geom import eval_mesh, fuse_from_splats
from .gs import render

_METRICS = {}


def _m():
    if not _METRICS:
        _METRICS["psnr"] = PeakSignalNoiseRatio(data_range=1.0).cuda()
        _METRICS["ssim"] = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        _METRICS["lpips"] = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()
    return _METRICS


@torch.no_grad()
def image_metrics(splats, scene, ids):
    """PSNR / SSIM / LPIPS on white background."""
    m = _m()
    white = torch.ones(1, 3, device="cuda")
    vals = {"psnr": [], "ssim": [], "lpips": []}
    for i in ids:
        rgb, _, _, _ = render(splats, scene.viewmats[i:i + 1], scene.K, scene.width, scene.height,
                              bg=white, mode="RGB")
        a = scene.alpha[i][..., None]
        gt = scene.rgb[i] * a + (1 - a)
        x, y = rgb.clamp(0, 1).permute(0, 3, 1, 2), gt[None].permute(0, 3, 1, 2)
        for k in vals:
            vals[k].append(m[k](x, y).item())
    return {k: sum(v) / len(v) for k, v in vals.items()}


def evaluate(splats, scene, gt_points, voxel_frac=0.002):
    test = scene.ids("test")
    out = image_metrics(splats, scene, test)
    mesh = fuse_from_splats(splats, scene, test, render, voxel=voxel_frac * scene.bbox_diag)
    out.update(eval_mesh(mesh, gt_points, scene.bbox_diag))
    out["num_gs"] = len(splats["means"])
    return out, mesh
