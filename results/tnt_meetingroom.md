# Supplementary: Tanks and Temples "Meetingroom" (real indoor scene with furniture)

Purpose: check the synthetic finding of Q3 (good rendering ≠ good geometry) on real captured data with a
laser-scanned ground truth. Supplementary experiment; its limits are listed below.

Data: 371 frames (1920×1080) of the official training image set (obtained from a Hugging Face mirror; the
official Google Drive links were rate-limited), GT point cloud / alignment / crop / reference COLMAP
trajectory from the official site. Evaluation: official T&T procedure (τ = 1 cm) ported to Open3D 0.18
(`scripts/tnt/eval_tnt.py`), with frame-index correspondences for the trajectory alignment.

| Step | Result |
|---|---|
| COLMAP (CPU, sequential matching) | 371 / 371 frames registered, 111k points, reprojection 0.67 px |
| gsplat 3DGS, 30k steps, every 8th frame held out | **PSNR 24.87**, SSIM 0.860, LPIPS 0.218 on held-out frames |
| Trajectory alignment to the reference COLMAP trajectory | camera-centre residual median **0.9 cm** (frame offset 0; ±1 frame gives 31 cm) |

## Geometry (3DGS rendered depth, back-projected; 5 mm voxel before evaluation)

| Final registration | Precision | Recall | F-score | Scale to GT |
|---|---|---|---|---|
| Official: ICP with scale | 0.035 | 0.187 | **0.059** | 1.574 |
| Rigid ICP (scale fixed from trajectory) | 0.028 | 0.118 | 0.045 | 1.475 |

- Median distance recon → GT is 18 cm: most reconstructed points are floaters, not surface points.
- The reconstruction still has 64 M points after 5 mm voxel downsampling (GT: 40 M) — the noise fills volume.

## What happened along the way (all recorded, nothing tuned against the GT F-score)

1. **Registration.** The official final ICP also optimises scale. On this noisy cloud it drifted from the
   trajectory scale 1.475 to 1.526 (first run) / 1.574 (final run). Checked with COLMAP sparse points (clean
   SfM points, independent of 3DGS): with the trajectory alignment their median distance to GT is 1.9 cm,
   with the official ICP transform 5.9 cm. So the trajectory alignment is the more trustworthy registration,
   but rigid ICP on this cloud was not stable either; both numbers are reported.
2. **Fusion.** TSDF fusion (first with a coarse ~1.5 cm voxel: F = 0.042; then with the 2DGS release settings
   for Meetingroom: 6 mm voxel, 2.4 cm truncation, 4.5 m depth) ran out of memory at 64 GB even when
   streaming frames: noisy 3DGS depth allocates ~0.5 GB of voxel blocks per frame. The final numbers use
   back-projected rendered depth (α > 0.5, depth < 4.5 m, every 2nd pixel, 3 mm voxel), which does no
   averaging, so floaters count fully against precision.
3. **Evaluation cost.** The unmodified official pipeline did not finish within 1.5 h on 91 M points; the
   final runs pre-downsample both clouds to τ/2 = 5 mm (the official P/R/F step downsamples to τ/2 anyway;
   only the ICP inputs change) and took ~1.5 h each.

## Interpretation and limits

- Qualitatively consistent with the synthetic Q3 result: acceptable rendering (PSNR 24.9) but very poor
  geometry straight from 3DGS depth, dominated by floaters.
- **Not comparable to published 3DGS F-scores**: those use TSDF fusion, which averages noise away; ours had
  to use unfiltered back-projection. The absolute F is therefore a lower bound for this model, not a
  statement that this 3DGS training is worse than published ones.
- Single scene. Fixing it properly (depth filtering / multi-view consistency, or a geometry-aware model such
  as 2DGS) is future work, not attempted here.
