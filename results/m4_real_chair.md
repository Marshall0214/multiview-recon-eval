# M4: real wooden chair (iPhone 12, printed A4 ArUco board, tape-measured dimensions)

Capture: 115 photos (HEIC → JPEG, 3024×4032, EXIF kept), two handheld rings, board flat on the floor.
Board printed at 99.6% (tape: pattern 169.3 × 229.2 mm vs 170 × 230 mm) → marker edge 49.81 mm.
Scene: glossy polished floor tiles (strong reflections), leather sofas nearby.

| Step | Result |
|---|---|
| ArUco detection | markers in 100 / 115 photos, all 12 in 54 |
| COLMAP (CPU, exhaustive) | 115 / 115 images registered |
| Metric alignment | 44 corners triangulated, RMS **0.66 mm** |

## Dimensions (mm)

| | Tape | v1 (pre-registered) | **v2 (post-hoc)** |
|---|---|---|---|
| Height | 804 | 855 (+51, +6.4%) | **819 (+15, +1.9%)** |
| Width | 584 | 736 (+152, +26%) | **556 (−28, −4.7%)** |
| Depth | 496 | 643 (+147, +30%) | **542 (+46, +9.4%)** |

v2 on the simulated capture (dev set, to check the fix does not break it): height −9 mm (−1.0%),
width −12 mm (−1.6%), depth −16 mm (−2.4%) (v1 there: −27 / +5 / +0 mm).

## What went wrong in v1, and the v2 fix (made after seeing these results)

![diagnosis](fig_real_chair_diag.png)

*Left: all points inside the camera ring (top view). Middle: the cluster v1 measured. Right: its side view.*

The chair itself is reconstructed well (back slats, armrests, seat, legs in the side view). Two failure modes
of a real room that the simulation did not have:

1. **Reflective floor.** 3DGS reconstructs the chair's reflection as geometry below the floor; v1's RANSAC
   floor (points within 8 cm of the board plane) was pulled ~3–5 cm down (the legs end at z ≈ 2–5 cm
   instead of 0), so height came out +51 mm. v2 keeps the ArUco board plane as the floor (RANSAC only
   within ±1 cm of it) — the board lies on the real floor and is aligned to 0.66 mm.
2. **Floor-level debris** 1.5–5 cm above the floor joined the legs and widened the footprint by ~15 cm.
   v2 drops points < 3 cm above the floor, takes the footprint between the 1st/99th percentiles along
   the min-area rectangle axes, and picks the *largest* cluster near the orbit centre (with the debris
   removed, a small blob under the seat was otherwise the *nearest* one).

Remaining depth error (+46 mm) — a possible, unverified cause: the backrest reclines, so from above it
overhangs the legs; the automatic footprint includes it, a tape measurement at seat level would not.

Honesty note: v2 was designed after seeing the real-chair errors, so it has only one real test object;
the v1 numbers are the pre-registered result.
