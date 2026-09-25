"""IG (v2) evaluated only on the held-out objects (table, bookcase); paired by object x seed."""
import pandas as pd

df = pd.read_csv("results/m2_rounds.csv")
h = df[df.object.isin(["table", "bookcase"])]
lines = ["# M2 held-out comparison for IG (v2)", "",
         "Objects not used while designing IG: table, bookcase (2 objects × 2 seeds = 4 paired runs per cell).", "",
         "| Init | Strategy | 15 views | 20 views | 25 views | 30 views | mean r1–4 |", "|---|---|---|---|---|---|---|"]
for init in ["biased", "uniform"]:
    for s in ["ours", "fps", "random", "ig"]:
        g = h[(h.init == init) & (h.strategy == s)]
        per = g.groupby("n_views")["fscore@1%"].mean()
        lines.append(f"| I-{init} | {s} | " + " | ".join(f"{per[n]:.3f}" for n in (15, 20, 25, 30))
                     + f" | {g[g['round'] > 0]['fscore@1%'].mean():.3f} |")
lines += ["", "Paired difference IG − FPS in F@1% after the first rescan (15 views), per object × seed:", ""]
for init in ["biased", "uniform"]:
    a = h[(h.init == init) & (h.n_views == 15)].pivot_table(index=["object", "seed"], columns="strategy", values="fscore@1%")
    d = (a["ig"] - a["fps"]).round(3)
    lines.append(f"- I-{init}: " + ", ".join(f"{o}/s{s}: {v:+.3f}" for (o, s), v in d.items())
                 + f" → mean {d.mean():+.3f}, IG better in {(d > 0).sum()}/{len(d)}")
open("results/m2_heldout.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
