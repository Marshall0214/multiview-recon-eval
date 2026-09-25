# M1: evaluation pipeline self-check

GT depth (Open3D ray casting on `mesh_gt.ply`) from all 200 candidate views, TSDF-fused
(voxel = 0.2% bbox diag), compared against the GT mesh. See `scripts/eval/check_gt_pipeline.py`.

| Object | Mask IoU (ray-cast vs Blender alpha) | Observable surface | F@1% vs all surface | F@1% vs observable GT | F@0.5% vs observable GT |
|---|---|---|---|---|---|
| B07J2YB486 (lounge chair) | 0.9999 | 70.9% | 0.868 (R=0.767) | 1.000 | 0.997 |

- Mask IoU ≈ 1 confirms the OpenCV/Blender camera conventions agree.
- Against uniformly sampled GT, even a perfect scan only reaches recall 0.77, because ~29% of the
  surface (underside, internal faces) is never visible from any view in the pool. Recall is therefore
  measured against **observable GT points** (visible from ≥1 view in the full cand+test pool).
