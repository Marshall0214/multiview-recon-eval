# M2: rescan results

Final round (30 views), mean ± std over objects × seeds (runs per cell: 6–6).

| Init | Strategy | F@1% | Recall@1% | Precision@1% | F@2% | PSNR | LPIPS |
|---|---|---|---|---|---|---|---|
| I-biased | S1+S2 (v1) | 0.707 ± 0.058 | 0.761 ± 0.109 | 0.671 ± 0.063 | 0.848 ± 0.049 | 30.08 | 0.080 |
| I-biased | FPS | 0.791 ± 0.139 | 0.788 ± 0.151 | 0.794 ± 0.128 | 0.907 ± 0.067 | 35.24 | 0.050 |
| I-biased | Random | 0.781 ± 0.132 | 0.794 ± 0.141 | 0.768 ± 0.124 | 0.893 ± 0.076 | 32.81 | 0.062 |
| I-biased | IG (v2) | 0.769 ± 0.125 | 0.776 ± 0.142 | 0.763 ± 0.110 | 0.901 ± 0.074 | 34.12 | 0.054 |
| I-uniform | S1+S2 (v1) | 0.781 ± 0.142 | 0.791 ± 0.148 | 0.772 ± 0.138 | 0.908 ± 0.066 | 34.32 | 0.056 |
| I-uniform | FPS | 0.786 ± 0.143 | 0.787 ± 0.151 | 0.788 ± 0.136 | 0.910 ± 0.064 | 35.90 | 0.048 |
| I-uniform | Random | 0.771 ± 0.138 | 0.792 ± 0.149 | 0.753 ± 0.133 | 0.898 ± 0.064 | 34.00 | 0.054 |
| I-uniform | IG (v2) | 0.763 ± 0.149 | 0.778 ± 0.148 | 0.749 ± 0.152 | 0.893 ± 0.075 | 34.66 | 0.050 |

## Per object (final round, F@1%, mean over seeds)

| Init | Object | S1+S2 (v1) | FPS | Random | IG (v2) |
|---|---|---|---|---|---|
| I-biased | chair | 0.734 | 0.714 | 0.719 | 0.707 |
| I-biased | table | 0.649 | 0.674 | 0.659 | 0.659 |
| I-biased | bookcase | 0.738 | 0.986 | 0.964 | 0.941 |
| I-uniform | chair | 0.690 | 0.696 | 0.698 | 0.658 |
| I-uniform | table | 0.673 | 0.677 | 0.657 | 0.658 |
| I-uniform | bookcase | 0.979 | 0.987 | 0.958 | 0.972 |

## F@1% per round (mean over objects × seeds) and mean over rescan rounds 1–4 (area-under-curve proxy)

| Init | Strategy | 10 views | 15 views | 20 views | 25 views | 30 views | mean r1–4 |
|---|---|---|---|---|---|---|---|
| I-biased | S1+S2 (v1) | 0.325 | 0.473 | 0.579 | 0.655 | 0.707 | 0.603 |
| I-biased | FPS | 0.327 | 0.617 | 0.752 | 0.782 | 0.791 | 0.736 |
| I-biased | Random | 0.326 | 0.556 | 0.705 | 0.763 | 0.781 | 0.701 |
| I-biased | IG (v2) | 0.324 | 0.668 | 0.742 | 0.748 | 0.769 | 0.732 |
| I-uniform | S1+S2 (v1) | 0.602 | 0.694 | 0.747 | 0.762 | 0.781 | 0.746 |
| I-uniform | FPS | 0.606 | 0.733 | 0.759 | 0.761 | 0.786 | 0.760 |
| I-uniform | Random | 0.605 | 0.715 | 0.732 | 0.768 | 0.771 | 0.747 |
| I-uniform | IG (v2) | 0.600 | 0.728 | 0.729 | 0.745 | 0.763 | 0.741 |

IG (v2) was designed after the v1 failure analysis using the chair only; table and bookcase are held out for it.


## Q1: signal vs true error (Spearman ρ over remaining candidates, per round)

`bad` = fraction of GT object pixels with depth error > 1% diag or alpha < 0.5 (higher = worse): a useful signal has **positive** ρ with bad and **negative** ρ with PSNR.

| Init | Signal | ρ vs bad (mean ± std) | ρ vs PSNR (mean ± std) | n |
|---|---|---|---|---|
| I-biased | S1 | -0.147 ± 0.204 | -0.162 ± 0.496 | 96 |
| I-biased | S2 | 0.084 ± 0.360 | -0.526 ± 0.389 | 96 |
| I-biased | S1+S2 | 0.056 ± 0.317 | -0.450 ± 0.256 | 96 |
| I-biased | IG (v2) | -0.147 ± 0.266 | -0.346 ± 0.287 | 24 |
| I-uniform | S1 | -0.195 ± 0.250 | -0.363 ± 0.279 | 96 |
| I-uniform | S2 | -0.084 ± 0.356 | -0.292 ± 0.372 | 96 |
| I-uniform | S1+S2 | -0.152 ± 0.290 | -0.423 ± 0.246 | 96 |
| I-uniform | IG (v2) | 0.010 ± 0.196 | -0.081 ± 0.274 | 24 |
