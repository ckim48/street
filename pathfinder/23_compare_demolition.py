"""Cross-city comparison of the Regime-2 demolition-by-HOLC-grade gradient.

Reads the per-city regime2_grade_uoi_{slug}.csv files and draws them side by side
so the key finding -- that highway/urban-renewal street demolition fell almost
entirely on the redlined (HOLC-D) fabric -- can be seen to REPLICATE across
independent cities (addressing the single-city limitation of the Detroit study).

Outputs: results/pathfinder/fig_demolition_compare.png
Usage: python pathfinder/23_compare_demolition.py [--cities detroit kansas_city]
"""
from __future__ import annotations

import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pf_style as S
from pf_common import CITIES, RES

GRADES = ["A", "B", "C", "D"]
GLABEL = {"A": "A\n(green)", "B": "B\n(blue)", "C": "C\n(yellow)", "D": "D\n(red)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["detroit", "kansas_city"])
    a = ap.parse_args()

    dfs = []
    for slug in a.cities:
        p = RES / f"regime2_grade_uoi_{slug}.csv"
        if p.exists():
            dfs.append((slug, pd.read_csv(p)))
    if not dfs:
        raise SystemExit("no regime2_grade_uoi_{slug}.csv found -- run 21_regime2 first")

    ymax = max(df["pct_demolished"].max() for _, df in dfs) * 1.18
    fig, axes = plt.subplots(1, len(dfs), figsize=(5.4 * len(dfs), 4.8),
                             sharey=True, squeeze=False)
    for ax, (slug, df) in zip(axes[0], dfs):
        cfg = CITIES[slug]
        vals, cols = [], []
        for g in GRADES:
            r = df[df.grade == g]
            vals.append(float(r.pct_demolished.values[0]) if len(r) else 0.0)
            cols.append(S.GRADE_COLOR[g])
        bars = ax.bar(range(len(GRADES)), vals, color=cols, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, v + ymax * 0.015, f"{v:.1f}%", ha="center",
                    fontsize=9, color=S.INK2)
        ax.set_xticks(range(len(GRADES)))
        ax.set_xticklabels([GLABEL[g] for g in GRADES], fontsize=9, color=S.INK2)
        py = cfg["build_start"] - 1
        S.title(ax, f"{cfg['city']} — {cfg['neighborhood']}\n{cfg['highway']} · "
                    f"{py}→post", fontsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("% of pre-highway street-km demolished", color=S.INK2)
    axes[0][0].set_ylim(0, ymax)
    fig.suptitle("Highway-era street demolition fell on the redlined (HOLC-D) "
                 "fabric — replicated across cities", fontsize=12, color=S.INK)
    fig.tight_layout()
    out = RES / "fig_demolition_compare.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)
    for slug, df in dfs:
        d = df[df.grade == "D"]["pct_demolished"].values
        print(f"  {slug}: D demolition = {d[0] if len(d) else 0:.1f}%")


if __name__ == "__main__":
    main()
