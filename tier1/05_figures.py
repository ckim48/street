"""Tier 1 - Step 6: mechanism & validation figures.

Produces the visual deliverables the guide asks for, from whatever stages have
run so far:

  fig_{slug}_boundary_map.png   dissolved HOLC grades + C-D/B-C boundary
                                segments coloured by barrier class (the guide's
                                per-city barrier-classification map).
  fig_{slug}_oi_by_decade.png   RD jump tau at the C-D / B-C frontier by decade
                                for the headline Oi metrics, with 95% CIs -> the
                                Oi-by-decade divergence trajectory (needs the
                                CHRONEX decade graphs; degrades to the single
                                'present' point otherwise).
  fig_tier1_rd_summary.png      cross-city RD forest plot for the latest decade.

Usage: python tier1/05_figures.py [--cities chicago ...]
"""
from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tier1_common import BND_DIR, CITIES, RD_DIR, RES, city_slugs

warnings.filterwarnings("ignore")

GRADE_FILL = {"A": "#76a865", "B": "#7cb5bd", "C": "#f0dd5c", "D": "#e05a5a"}
BARRIER_COLOR = {"street": "#333333", "rail": "#8c564b", "water": "#1f77b4",
                 "freeway": "#d62728", "harbor": "#17becf", "ambiguous": "#999999",
                 "unclassified": "#999999"}
HEADLINE = ["intersection_density", "median_block_length_ft", "link_node_ratio"]


def _decade_key(d):
    return 9999 if str(d) == "present" else int(d)


def boundary_map(slug):
    gp = BND_DIR / f"{slug}_grades.gpkg"; bp = BND_DIR / f"{slug}_boundaries.gpkg"
    if not (gp.exists() and bp.exists()):
        return
    grades = gpd.read_file(gp); segs = gpd.read_file(bp)
    fig, ax = plt.subplots(figsize=(9, 9))
    for _, r in grades.iterrows():
        gpd.GeoSeries([r.geometry]).plot(ax=ax, color=GRADE_FILL.get(r.grade, "#ccc"),
                                         alpha=0.45, edgecolor="white", lw=0.3)
    for bcls, sub in segs.groupby("barrier"):
        sub.plot(ax=ax, color=BARRIER_COLOR.get(bcls, "#999"), lw=1.6,
                 label=f"{bcls} ({sub.length.sum()/1000:.0f} km)")
    from matplotlib.patches import Patch
    handles = [Patch(color=GRADE_FILL[g], alpha=0.45, label=f"grade {g}")
               for g in "ABCD" if g in set(grades.grade)]
    leg1 = ax.legend(handles=handles, loc="upper left", fontsize=8, title="HOLC grade")
    ax.add_artist(leg1)
    ax.legend(loc="lower right", fontsize=8, title="boundary segment")
    ax.set_title(f"{CITIES[slug]['city']} — HOLC grades & classified C-D/B-C "
                 f"boundaries\n{CITIES[slug]['role']}", fontsize=11)
    ax.set_axis_off(); fig.tight_layout()
    fig.savefig(RES / f"fig_{slug}_boundary_map.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig_{slug}_boundary_map.png")


def oi_by_decade(slug):
    p = RD_DIR / f"{slug}_rd.csv"
    if not p.exists():
        return
    rd = pd.read_csv(p)
    rd["dk"] = rd["decade"].map(_decade_key)
    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(5 * len(HEADLINE), 4.2),
                             squeeze=False)
    for j, metric in enumerate(HEADLINE):
        ax = axes[0][j]
        for pair, c in [("C-D", "#e05a5a"), ("B-C", "#f0a020")]:
            d = rd[(rd.metric == metric) & (rd.pair == pair)].sort_values("dk")
            if not len(d):
                continue
            ax.errorbar(d.dk, d.tau, yerr=1.96 * d.se, marker="o", color=c,
                        capsize=3, lw=1.5, label=pair)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("decade")
        if j == 0:
            ax.set_ylabel("RD jump τ (lower-grade − higher-grade)")
        ax.legend(fontsize=8)
    fig.suptitle(f"{CITIES[slug]['city']} — Oi discontinuity at the HOLC frontier "
                 f"by decade", fontsize=12)
    fig.tight_layout()
    fig.savefig(RES / f"fig_{slug}_oi_by_decade.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig_{slug}_oi_by_decade.png")


def rd_summary(cities):
    p = RES / "rd_all.csv"
    if not p.exists():
        return
    rd = pd.read_csv(p)
    rd["dk"] = rd["decade"].map(_decade_key)
    latest = rd.sort_values("dk").groupby(["slug", "pair", "metric"]).tail(1)
    sub = latest[(latest.pair == "C-D") &
                 (latest.metric == "intersection_density")].copy()
    sub = sub[sub.slug.isin(cities)].sort_values("tau")
    if not len(sub):
        return
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(sub) + 1.5))
    y = np.arange(len(sub))
    ax.errorbar(sub.tau, y, xerr=1.96 * sub.se, fmt="o", color="#4C72B0", capsize=3)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([CITIES[s]["city"] for s in sub.slug])
    ax.set_xlabel("C-D RD jump in intersection density (lower − higher grade)")
    ax.set_title("Tier 1 — HOLC C-D discontinuity in intersection density\n"
                 "(latest available decade per city)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / "fig_tier1_rd_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig_tier1_rd_summary.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=city_slugs())
    args = ap.parse_args()
    for slug in args.cities:
        print(f"[{slug}]")
        boundary_map(slug)
        oi_by_decade(slug)
    rd_summary(args.cities)
    print(f"\nfigures -> {RES}")


if __name__ == "__main__":
    main()
