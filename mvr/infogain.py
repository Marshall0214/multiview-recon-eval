"""v2 signal (post-hoc, after the v1 failure analysis): volumetric information gain.

Classic robotics NBV primitive, computed from the current 3DGS reconstruction only (no GT):

  voxel states  -- inside the visual hull, a voxel is "known" once some training view sees it
                   unoccluded: it projects into the image and lies in front of, or on, the surface
                   rendered from that view. Hull voxels never seen that way are "unknown".
  gain(v)       -- unknown voxels that candidate v would see before its rays hit the rendered surface.
  selection     -- greedy set cover: after picking a view, its unknown voxels become known.

Why v1's S1 failed: with few views the hull is loose, so "hull pixel but low alpha" measures hull
looseness from that direction, not missing geometry. Reasoning in 3D with depth ordering fixes that.
"""

import torch

from .gs import render
from .hull import carve, project


@torch.no_grad()
def _visible(pts, splats, scene, view_ids, margin):
    """[len(view_ids), N] bool: voxel projects into view and is not behind the rendered surface."""
    out = []
    for i in view_ids:
        vm = scene.viewmats[i:i + 1]
        _, alpha, depth, _ = render(splats, vm, scene.K, scene.width, scene.height)
        uv, z = project(pts, vm, scene.K)
        u, v, z = uv[0, :, 0].round().long(), uv[0, :, 1].round().long(), z[0]
        inside = (z > 0) & (u >= 0) & (u < scene.width) & (v >= 0) & (v < scene.height)
        uc, vc = u.clamp(0, scene.width - 1), v.clamp(0, scene.height - 1)
        d = depth[0][vc, uc]
        empty = alpha[0][vc, uc] < 0.5
        out.append(inside & (empty | (z <= d + margin)))
    return torch.stack(out)


@torch.no_grad()
def unknown_voxels(splats, scene, train_ids, res=96):
    centers, voxel = carve(scene, train_ids, res=res)
    seen = _visible(centers, splats, scene, train_ids, margin=voxel).any(0)
    return centers[~seen], voxel


@torch.no_grad()
def gain_matrix(splats, scene, train_ids, cand_ids, res=96):
    """[len(cand_ids), n_unknown] visibility of unknown voxels from each candidate."""
    unk, voxel = unknown_voxels(splats, scene, train_ids, res)
    if len(unk) == 0:
        return torch.zeros(len(cand_ids), 0, dtype=torch.bool, device="cuda")
    return _visible(unk, splats, scene, cand_ids, margin=voxel)


def select_greedy(vis, n):
    """Greedy set cover on the visibility matrix; returns row indices and their marginal gains."""
    covered = torch.zeros(vis.shape[1], dtype=torch.bool, device=vis.device)
    picks, gains = [], []
    for _ in range(n):
        g = (vis & ~covered).sum(1)
        if picks:
            g[picks] = -1
        j = int(g.argmax())
        picks.append(j)
        gains.append(int(g[j]))
        covered |= vis[j]
    return picks, gains
