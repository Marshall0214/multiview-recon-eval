"""Load a rendered scene (scripts/render/render_abo.py output) into GPU tensors."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import imageio.v2 as imageio
import numpy as np
import torch


@dataclass
class Scene:
    root: Path
    names: List[str]          # view names, e.g. cand_000
    splits: List[str]         # "cand" / "test"
    c2w: torch.Tensor         # [V, 4, 4] OpenCV convention
    K: torch.Tensor           # [3, 3]
    width: int
    height: int
    rgb: torch.Tensor         # [V, H, W, 3] in [0, 1], straight (not premultiplied)
    alpha: torch.Tensor       # [V, H, W]
    bbox_diag: float
    radius: float

    def ids(self, split):
        return [i for i, s in enumerate(self.splits) if s == split]

    @property
    def viewmats(self):
        return torch.linalg.inv(self.c2w)

    @property
    def centers(self):
        return self.c2w[:, :3, 3]


def load_scene(root, device="cuda", downscale=1):
    root = Path(root)
    meta = json.loads((root / "cameras.json").read_text())
    rgbs, alphas = [], []
    for f in meta["frames"]:
        im = imageio.imread(root / "images" / f"{f['name']}.png").astype(np.float32) / 255.0
        if downscale > 1:
            im = im[::downscale, ::downscale]
        rgbs.append(im[..., :3])
        alphas.append(im[..., 3])
    K = torch.tensor([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]],
                     dtype=torch.float32)
    K[:2] /= downscale
    return Scene(
        root=root,
        names=[f["name"] for f in meta["frames"]],
        splits=[f["split"] for f in meta["frames"]],
        c2w=torch.tensor(np.array([f["c2w"] for f in meta["frames"]]), dtype=torch.float32, device=device),
        K=K.to(device),
        width=meta["width"] // downscale,
        height=meta["height"] // downscale,
        rgb=torch.from_numpy(np.stack(rgbs)).to(device),
        alpha=torch.from_numpy(np.stack(alphas)).to(device),
        bbox_diag=meta["bbox_diag"],
        radius=meta["radius"],
    )
