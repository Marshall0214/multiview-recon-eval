# M0: gsplat 3DGS on Mip-NeRF 360

Reference = gsplat-Xk rows in `third_party/gsplat/docs/source/tests/eval.rst` (TITAN RTX).
Ours = RTX 4090 D, gsplat 1.5.3, `simple_trainer.py default`, test_every=8.

| Scene | Steps | PSNR (ours / ref) | SSIM (ours / ref) | LPIPS (ours / ref) | #GS | Train time |
|---|---|---|---|---|---|---|
| room | 7k | 29.88 / 29.21 | 0.908 / 0.893 | 0.192 / 0.217 | 1.46M | 2.5 min |
| room | 30k | 31.63 / 31.36 | 0.929 / 0.918 | 0.149 / 0.164 | 2.22M | 15.9 min |
