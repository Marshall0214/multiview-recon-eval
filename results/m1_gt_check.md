# M1: evaluation pipeline self-check

GT depth (Open3D ray casting on `mesh_gt.ply`) from all 200 candidate views, TSDF-fused
(voxel = 0.2% bbox diag), compared against the GT mesh. See `scripts/eval/check_gt_pipeline.py`.

| Object | Mask IoU (ray-cast vs Blender alpha) | Observable surface | F@1% vs all surface | F@1% vs observable GT | F@0.5% vs observable GT |
|---|---|---|---|---|---|
| B07J2YB486 (lounge chair) | 0.9999 | 70.9% | 0.868 (R=0.767) | 1.000 | 0.997 |
| B07B82PXCW (drop-leaf table) | 0.9999 | 92.9% | 0.987 (R=0.974) | 1.000 | 0.999 |
| B07HSCJZQM (ladder bookcase) | 0.9996 | 89.5% | 1.000 (R=1.000) | 1.000 | 0.999 |

- Mask IoU ≈ 1 confirms the OpenCV/Blender camera conventions agree.
- Against uniformly sampled GT, even a perfect scan only reaches recall 0.77, because ~29% of the
  surface (underside, internal faces) is never visible from any view in the pool. Recall is therefore
  measured against **observable GT points** (visible from ≥1 view in the full cand+test pool).

## Dense-view upper bound (chair, all 200 candidate views, 7k steps)

| PSNR | SSIM | LPIPS | Precision@1% | Recall@1% | F@1% | F@2% |
|---|---|---|---|---|---|---|
| 41.06 | 0.984 | 0.062 | 0.796 | 0.817 | 0.806 | 0.918 |

Near-perfect rendering but only 0.81 F-score. Depth error on 20 test views (object pixels with alpha > 0.5):
median signed error **+0.27% of bbox diag** (no systematic offset, so not a convention bug), but
**32%** of pixels are off by more than 1% (p90 |err| = 2.5%), biased *behind* the surface. This is the
known expected-depth bias of volumetric 3DGS (Gaussians spread through a shell behind the surface), and
the first evidence for Q3: rendering quality does not imply geometric accuracy.
