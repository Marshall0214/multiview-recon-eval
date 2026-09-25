"""Minimal 3DGS trainer / renderer on top of gsplat (object-level, known poses, masked images).

Follows gsplat's simple_trainer defaults (lrs, L1 + 0.2 SSIM, DefaultStrategy densification),
scaled to a short schedule. Differences, all deliberate:
  * init: random points inside the visual hull (no SfM on synthetic data)
  * random background colour each step, so alpha is forced to be correct (S1 relies on alpha)
"""

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

from .hull import carve

try:
    from fused_ssim import fused_ssim
except ImportError:  # pragma: no cover
    fused_ssim = None


@dataclass
class TrainConfig:
    steps: int = 7000
    init_pts: int = 50_000
    sh_degree: int = 3
    sh_interval: int = 1000
    ssim_lambda: float = 0.2
    refine_stop: int = 5000
    hull_res: int = 128
    seed: int = 0


def _sh0(rgb):
    return (rgb - 0.5) / 0.28209479177387814


def init_splats(scene, train_ids, cfg):
    g = torch.Generator(device="cuda").manual_seed(cfg.seed)
    centers, voxel = carve(scene, train_ids, res=cfg.hull_res)
    idx = torch.randint(len(centers), (cfg.init_pts,), device="cuda", generator=g)
    pts = centers[idx] + (torch.rand((cfg.init_pts, 3), device="cuda", generator=g) - 0.5) * voxel
    n = len(pts)
    scales = torch.full((n, 3), math.log(voxel), device="cuda")
    colors = torch.zeros((n, (cfg.sh_degree + 1) ** 2, 3), device="cuda")
    colors[:, 0] = _sh0(torch.full((n, 3), 0.5, device="cuda"))
    params = {
        "means": (pts, 1.6e-4 * scene.bbox_diag),
        "scales": (scales, 5e-3),
        "quats": (torch.rand((n, 4), device="cuda", generator=g), 1e-3),
        "opacities": (torch.logit(torch.full((n,), 0.1, device="cuda")), 5e-2),
        "sh0": (colors[:, :1], 2.5e-3),
        "shN": (colors[:, 1:], 2.5e-3 / 20),
    }
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, (v, _) in params.items()})
    optims = {k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15)
              for k, (_, lr) in params.items()}
    return splats, optims


def render(splats, viewmats, K, width, height, bg=None, sh_degree=3, colors=None, mode="RGB+ED"):
    """Returns (rgb [C,H,W,D], alpha [C,H,W], depth [C,H,W] or None, info)."""
    C = len(viewmats)
    if colors is None:
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    else:
        sh_degree = None
    out, alpha, info = rasterization(
        means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]), colors=colors,
        viewmats=viewmats, Ks=K[None].expand(C, 3, 3), width=width, height=height,
        sh_degree=sh_degree, backgrounds=bg, render_mode=mode, packed=False,  # packed=True + backgrounds trips a shape assert in gsplat 1.5.3
    )
    if mode.endswith("ED"):
        return out[..., :-1], alpha[..., 0], out[..., -1], info
    return out, alpha[..., 0], None, info


def train(scene, train_ids, cfg=TrainConfig(), log_every=0):
    torch.manual_seed(cfg.seed)
    splats, optims = init_splats(scene, train_ids, cfg)
    strategy = DefaultStrategy(refine_stop_iter=cfg.refine_stop, verbose=False)
    strategy.check_sanity(splats, optims)
    state = strategy.initialize_state(scene_scale=scene.bbox_diag)
    sched = torch.optim.lr_scheduler.ExponentialLR(optims["means"], gamma=0.01 ** (1.0 / cfg.steps))
    g = torch.Generator().manual_seed(cfg.seed)
    ids = torch.tensor(train_ids)
    t0 = time.time()
    for step in range(cfg.steps):
        i = int(ids[torch.randint(len(ids), (1,), generator=g)])
        bg = torch.rand(3, device="cuda")
        gt = scene.rgb[i] * scene.alpha[i][..., None] + bg * (1 - scene.alpha[i][..., None])
        rgb, _, _, info = render(splats, scene.viewmats[i:i + 1], scene.K, scene.width, scene.height,
                                 bg=bg[None], sh_degree=min(step // cfg.sh_interval, cfg.sh_degree),
                                 mode="RGB")
        strategy.step_pre_backward(splats, optims, state, step, info)
        l1 = F.l1_loss(rgb[0], gt)
        if fused_ssim is not None:
            ssim = fused_ssim(rgb.permute(0, 3, 1, 2), gt[None].permute(0, 3, 1, 2), padding="valid")
        else:
            ssim = torch.tensor(1.0, device="cuda")
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim)
        loss.backward()
        for o in optims.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(splats, optims, state, step, info, packed=False)
        if log_every and step % log_every == 0:
            print(f"step {step:5d} loss {loss.item():.4f} n_gs {len(splats['means'])}")
    torch.cuda.synchronize()
    return splats, time.time() - t0
