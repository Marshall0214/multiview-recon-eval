# M2 held-out comparison for IG (v2)

Objects not used while designing IG: table, bookcase (2 objects × 2 seeds = 4 paired runs per cell).

| Init | Strategy | 15 views | 20 views | 25 views | 30 views | mean r1–4 |
|---|---|---|---|---|---|---|
| I-biased | ours | 0.463 | 0.559 | 0.633 | 0.694 | 0.587 |
| I-biased | fps | 0.647 | 0.792 | 0.833 | 0.830 | 0.776 |
| I-biased | random | 0.583 | 0.735 | 0.797 | 0.811 | 0.732 |
| I-biased | ig | 0.683 | 0.780 | 0.803 | 0.800 | 0.766 |
| I-uniform | ours | 0.743 | 0.797 | 0.812 | 0.826 | 0.795 |
| I-uniform | fps | 0.771 | 0.808 | 0.811 | 0.832 | 0.805 |
| I-uniform | random | 0.768 | 0.781 | 0.801 | 0.808 | 0.789 |
| I-uniform | ig | 0.807 | 0.806 | 0.807 | 0.815 | 0.809 |

Paired difference IG − FPS in F@1% after the first rescan (15 views), per object × seed:

- I-biased: bookcase/s0: -0.026, bookcase/s1: +0.063, table/s0: +0.016, table/s1: +0.089 → mean +0.036, IG better in 3/4
- I-uniform: bookcase/s0: +0.009, bookcase/s1: +0.023, table/s0: +0.064, table/s1: +0.045 → mean +0.035, IG better in 4/4
