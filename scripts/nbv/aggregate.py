"""Aggregate M2 runs into tables (results/m2_summary.md, results/m2_rounds.csv) and figures.

Usage: python scripts/nbv/aggregate.py --runs /root/outputs/nbv --out results
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

OBJECTS = {"B07J2YB486": "chair", "B07B82PXCW": "table", "B07HSCJZQM": "bookcase"}
STRATS = ["ours", "fps", "random", "ig"]
LABEL = {"ours": "S1+S2 (v1)", "fps": "FPS", "random": "Random", "ig": "IG (v2)"}
# categorical slots 1-4 of the reference palette, validated (scripts/validate_palette.js, light)
COLOR = {"ours": "#2a78d6", "fps": "#eb6834", "random": "#1baf7a", "ig": "#eda100"}
MARKER = {"ours": "o", "fps": "s", "random": "^", "ig": "D"}  # secondary encoding: identity is never colour alone
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_rounds(runs):
    rows = []
    for f in runs.glob("*/*/*/seed*/round*.json"):
        obj, init, strat, seed = f.parts[-5:-1]
        r = json.loads(f.read_text())
        r.pop("views", None)
        rows.append({"object": OBJECTS.get(obj, obj), "init": init, "strategy": strat,
                     "seed": int(seed[4:]), **r})
    return pd.DataFrame(rows)


def load_correlations(runs):
    rows = []
    for f in runs.glob("*/*/*/seed*/round*_candidates.npz"):
        obj, init, strat, seed = f.parts[-5:-1]
        z = np.load(f)
        row = {"object": OBJECTS.get(obj, obj), "init": init, "strategy": strat, "seed": int(seed[4:]),
               "round": int(f.name[5:].split("_")[0])}
        for s in ("S1", "S2", "score", "IG"):
            if s not in z:
                continue
            row[f"{s}~bad"] = spearmanr(z[s], z["bad"])[0]
            row[f"{s}~psnr"] = spearmanr(z[s], z["psnr"])[0]
        rows.append(row)
    return pd.DataFrame(rows)


def pm(x):
    return f"{x.mean():.3f} ± {x.std(ddof=0):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def curves(df, metric, ylabel, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True, facecolor=SURFACE)
    for ax, init in zip(axes, ["biased", "uniform"]):
        ax.set_facecolor(SURFACE)
        sub = df[df.init == init]
        ends = []
        for s in STRATS:
            g = sub[sub.strategy == s].groupby("n_views")[metric]
            if not len(g):
                continue
            m, sd = g.mean(), g.std(ddof=0)
            ax.fill_between(m.index, m - sd, m + sd, color=COLOR[s], alpha=0.12, linewidth=0)
            ax.plot(m.index, m.values, color=COLOR[s], lw=2, marker=MARKER[s], ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABEL[s])
            ends.append([m.values[-1], m.index[-1], LABEL[s]])
        # direct end labels, nudged apart vertically so they never overlap
        lo, hi = ax.get_ylim()
        gap = 0.06 * (hi - lo)
        ends.sort()
        orig_mean = np.mean([e[0] for e in ends]) if ends else 0
        for k in range(1, len(ends)):
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
        shift = orig_mean - np.mean([e[0] for e in ends]) if ends else 0  # re-centre the pushed stack
        for e in ends:
            e[0] += shift
        for y, x, text in ends:
            ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK2, annotation_clip=False)
        ax.set_title(f"I-{init} initial views", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("training views", color=INK2, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.set_xlim(right=ax.get_xlim()[1] + 4)
    axes[0].set_ylabel(ylabel, color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("/root/outputs/nbv"))
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_rounds(args.runs)
    df.sort_values(["object", "init", "strategy", "seed", "round"]).to_csv(args.out / "m2_rounds.csv", index=False)
    corr = load_correlations(args.runs)
    corr.to_csv(args.out / "m2_correlations.csv", index=False)

    last = df[df["round"] == df["round"].max()]
    n_runs = last.groupby(["init", "strategy"]).size()
    lines = ["# M2: rescan results", "",
             f"Final round ({int(last.n_views.max())} views), mean ± std over objects × seeds "
             f"(runs per cell: {n_runs.min()}–{n_runs.max()}).", "",
             "| Init | Strategy | F@1% | Recall@1% | Precision@1% | F@2% | PSNR | LPIPS |",
             "|---|---|---|---|---|---|---|---|"]
    for init in ["biased", "uniform"]:
        for s in STRATS:
            g = last[(last.init == init) & (last.strategy == s)]
            if len(g):
                lines.append(f"| I-{init} | {LABEL[s]} | {pm(g['fscore@1%'])} | {pm(g['recall@1%'])} | "
                             f"{pm(g['precision@1%'])} | {pm(g['fscore@2%'])} | {g.psnr.mean():.2f} | {g.lpips.mean():.3f} |")

    lines += ["", "## Per object (final round, F@1%, mean over seeds)", "",
              "| Init | Object | " + " | ".join(LABEL[s] for s in STRATS) + " |", "|---|---|" + "---|" * len(STRATS)]
    for init in ["biased", "uniform"]:
        for obj in OBJECTS.values():
            g = last[(last.init == init) & (last.object == obj)]
            if len(g):
                vals = [g[g.strategy == s]["fscore@1%"].mean() for s in STRATS]
                lines.append(f"| I-{init} | {obj} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    lines += ["", "## F@1% per round (mean over objects × seeds) and mean over rescan rounds 1–4 (area-under-curve proxy)", "",
              "| Init | Strategy | " + " | ".join(f"{int(n)} views" for n in sorted(df.n_views.unique())) + " | mean r1–4 |",
              "|---|---|" + "---|" * (df.n_views.nunique() + 1)]
    for init in ["biased", "uniform"]:
        for s in STRATS:
            g = df[(df.init == init) & (df.strategy == s)]
            if len(g):
                per = g.groupby("n_views")["fscore@1%"].mean()
                auc = g[g["round"] > 0]["fscore@1%"].mean()
                lines.append(f"| I-{init} | {LABEL[s]} | " + " | ".join(f"{v:.3f}" for v in per.values) + f" | {auc:.3f} |")
    lines += ["", "IG (v2) was designed after the v1 failure analysis using the chair only; table and bookcase are held out for it.", ""]

    lines += ["", "## Q1: signal vs true error (Spearman ρ over remaining candidates, per round)", "",
              "`bad` = fraction of GT object pixels with depth error > 1% diag or alpha < 0.5 (higher = worse): "
              "a useful signal has **positive** ρ with bad and **negative** ρ with PSNR.", "",
              "| Init | Signal | ρ vs bad (mean ± std) | ρ vs PSNR (mean ± std) | n |", "|---|---|---|---|---|"]
    for init in ["biased", "uniform"]:
        c = corr[corr.init == init]
        for s in ("S1", "S2", "score", "IG"):
            name = {"score": "S1+S2", "IG": "IG (v2)"}.get(s, s)
            cc = c.dropna(subset=[f"{s}~bad"]) if f"{s}~bad" in c else c.iloc[:0]
            if len(cc):
                lines.append(f"| I-{init} | {name} | {pm(cc[f'{s}~bad'])} | {pm(cc[f'{s}~psnr'])} | {len(cc)} |")

    (args.out / "m2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    curves(df, "fscore@1%", "F-score @ 1% diag", args.out / "fig_nbv_fscore.png")
    curves(df, "recall@1%", "Recall (completeness) @ 1%", args.out / "fig_nbv_recall.png")
    curves(df, "psnr", "PSNR (test views)", args.out / "fig_nbv_psnr.png")


if __name__ == "__main__":
    main()
