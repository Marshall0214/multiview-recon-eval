# M4 pipeline validation on a simulated handheld capture

Before real photos: the full real-data pipeline (`scripts/real/run_real.py`) run on a Blender-rendered
capture that follows `docs/capture_guide.md` (ABO chair B07J2YB486 on a textured floor, printed A4
ArUco page next to it, 2 handheld rings x 45 photos, 1600x1200, ~65 deg HFOV, pose jitter). Only the
JPEGs go in; poses, scale and floor are recovered by COLMAP + ArUco exactly as for real data.

| Step | Result |
|---|---|
| COLMAP (CPU SIFT, exhaustive) | 90 / 90 images registered |
| ArUco metric alignment | 48 corners triangulated, RMS 1.15 mm |

| Dimension | GT (mm) | Reconstructed (mm) | Error (mm) | Error (%) |
|---|---|---|---|---|
| Height | 858.8 | 831.6 | −27.2 | −3.2 |
| Width | 769.0 | 773.8 | +4.8 | +0.6 |
| Depth | 678.3 | 679.8 | +1.5 | +0.2 |

Issues found and fixed on this dev capture (the real capture is the test set, no tuning there):
1. TSDF fused the whole 8 m floor and the far background (3DGS models them too) -> fusion limited to
   1.7x the camera-ring radius, mesh cropped to the inside of the ring before sampling.
2. The A4 board pins the floor only locally -> floor refined by RANSAC on near-floor points.
3. Height is biased low: the thin top edge of the chair back is eroded (edge pixels have alpha < 0.5
   in most views, so they are masked out before fusion); the tallest reconstructed point is 22 mm
   below GT. Height uses the 99.9th percentile (was 99.5th, −37 mm).
