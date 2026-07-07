"""Tier 1 - Step 6 (extras): the two deliverables the guide's Outputs list still
wants beyond the per-city boundary/Oi/RD figures:

  fig_tier1_mechanisms.png              the mechanism figure SET — one panel per
                                        deep-dive city (Chicago benchmark,
                                        Detroit barrier, Atlanta growth, LA
                                        freeway), each showing that city's C-D
                                        discontinuity signature across all six
                                        Oi metrics (t = tau/se, latest decade).
  fig_tier1_samecorner_validation.png   same-place 1940 vs 2020 street-network
                                        panels (Guide Step 4). Uses the CHRONEX
                                        decade graphs cropped to the HOLC
                                        footprint — the automatable stand-in for
                                        the manual topoView/Sanborn scan pairs.

Usage: python tier1/06_mechanism_validation.py
"""
from __future__ import annotations
import warnings

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import pandas as pd

from tier1_common import BND_DIR, CITIES, GRAPH_DIR, RES

warnings.filterwarnings("ignore")

METRICS = ["intersection_density", "median_block_length_ft", "link_node_ratio",
           "connected_node_ratio", "walking_circuity", "pedshed_reach"]
MECH = [("chicago", "Benchmark — clean C-D contrast on a uniform grid"),
        ("detroit", "Barrier — physical barriers & post-1950 divergence"),
        ("atlanta", "Growth — Sunbelt widening-gap setting"),
        ("los_angeles", "Freeway — freeways as post-treatment barriers")]


def _decade_key(d):
    return 9999 if str(d) == "present" else int(d)


# ---------------------------------------------------------------- mechanisms
def mechanisms():
    rd = pd.read_csv(RES / "rd_all.csv")
    rd["dk"] = rd["decade"].map(_decade_key)
    latest = rd.sort_values("dk").groupby(["slug", "pair", "metric"]).tail(1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (slug, tagline) in zip(axes.ravel(), MECH):
        sub = latest[(latest.slug == slug) & (latest.pair == "C-D")]
        sub = sub.set_index("metric").reindex(METRICS)
        t = (sub.tau / sub.se).values
        y = np.arange(len(METRICS))[::-1]
        colors = ["#C44E52" if abs(v) >= 1.96 else "#B7C4D8" for v in np.nan_to_num(t)]
        ax.barh(y, np.nan_to_num(t), color=colors, edgecolor="#333", lw=0.4)
        ax.axvline(0, color="k", lw=0.8)
        for x in (-1.96, 1.96):
            ax.axvline(x, color="green", ls=":", lw=1)
        ax.set_yticks(y); ax.set_yticklabels(METRICS, fontsize=8)
        dec = int(sub.dk.dropna().max()) if sub.dk.notna().any() else "?"
        n = int(sub.n.dropna().max()) if sub.n.notna().any() else 0
        ax.set_title(f"{CITIES[slug]['city']} — {tagline}\n"
                     f"C-D discontinuity signature (decade {dec}, n≈{n})", fontsize=10)
        ax.set_xlabel("RD t-statistic  (τ / SE);  |t|≥1.96 = significant")
    fig.suptitle("Tier 1 mechanism set — per-city C-D discontinuity across the six Oi metrics\n"
                 "red = significant jump at the HOLC frontier (lower − higher grade)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(RES / "fig_tier1_mechanisms.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_tier1_mechanisms.png")


# ----------------------------------------------------- same-corner validation
def _ekey(a, b):
    """Unordered edge key from endpoint coordinates (rounded to 1 m)."""
    return frozenset({(round(a[0]), round(a[1])), (round(b[0]), round(b[1]))})


def _load_segs(slug, decade):
    """Return (segments, keyset). Edges are keyed by ROUNDED endpoint COORDINATES,
    not node IDs — the CHRONEX decade graphml files reuse node IDs for different
    locations, so an ID-based diff is meaningless; coordinates are the geometry."""
    p = GRAPH_DIR / f"{slug}_{decade}.graphml"
    if not p.exists():
        return None, None
    G = nx.read_graphml(p)
    xy = {n: (float(d["x"]), float(d["y"])) for n, d in G.nodes(data=True)
          if "x" in d and "y" in d}
    segs, keys = [], set()
    for u, v in G.edges():
        if u in xy and v in xy:
            a, b = xy[u], xy[v]
            segs.append((a, b)); keys.add(_ekey(a, b))
    return segs, keys


def _in(seg, bbox):
    x0, y0, x1, y1 = bbox
    (ax, ay), (bx, by) = seg
    return x0 <= ax <= x1 and y0 <= ay <= y1 and x0 <= bx <= x1 and y0 <= by <= y1


def _segkey(seg):
    return _ekey(seg[0], seg[1])


def samecorner(cities=("chicago", "philadelphia", "atlanta"), half=2500.0):
    """1940 vs 2020 at the SAME window, chosen where post-1940 growth actually is,
    with streets added since 1940 (coordinate-based diff) drawn in red."""
    fig, axes = plt.subplots(len(cities), 2,
                             figsize=(9.4, 4.7 * len(cities)), squeeze=False)
    for i, slug in enumerate(cities):
        s40, k40 = _load_segs(slug, 1940)
        s20, k20 = _load_segs(slug, 2020)
        if s20 is None:
            continue
        added = [s for s in s20 if _segkey(s) not in k40]     # true new streets
        old = [s for s in s20 if _segkey(s) in k40]
        pts = np.array([p for s in added for p in s]) if added else np.empty((0, 2))
        if len(pts) >= 20:
            cx, cy = np.median(pts, axis=0)
        else:
            g = gpd.read_file(BND_DIR / f"{slug}_grades.gpkg")
            cx, cy = g.union_all().centroid.coords[0]
        bbox = (cx - half, cy - half, cx + half, cy + half)

        w40 = [s for s in s40 if _in(s, bbox)]
        w_old = [s for s in old if _in(s, bbox)]
        w_new = [s for s in added if _in(s, bbox)]
        for j, (title, layers) in enumerate([
                (f"{CITIES[slug]['city']} — 1940  ({len(w40)} edges)",
                 [(w40, "#1f3b63", 0.5)]),
                (f"{CITIES[slug]['city']} — 2020  (+{len(w_new)} new since 1940 in view)",
                 [(w_old, "#b7b7b7", 0.5), (w_new, "#d62728", 1.3)])]):
            ax = axes[i][j]
            for segs, c, lw in layers:
                if segs:
                    ax.add_collection(LineCollection([list(s) for s in segs], colors=c, linewidths=lw))
            ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=10)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color="#b0b0b0", label="street present in 1940"),
                        Line2D([0], [0], color="#d62728", label="street added 1940→2020")],
               loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Tier 1 same-place validation — street network 1940 vs 2020, identical "
                 f"{2*half/1000:.0f} km window centred on where post-1940 growth occurs\n"
                 "(CHRONEX-US decade graphs; automatable stand-in for the manual topoView/Sanborn scan pairs)",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(RES / "fig_tier1_samecorner_validation.png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_tier1_samecorner_validation.png")


if __name__ == "__main__":
    mechanisms()
    samecorner()
    print(f"\n-> {RES}")
