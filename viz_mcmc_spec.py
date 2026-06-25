"""Visualize the Stage-5 spec MCMC optimal-network results.

Reads data/outputs/sampler_spec/{summary.json, *.pkl} and emits, under
results/mcmc_spec/:
  dtf_table.csv                per-tract distance-to-frontier + before/after
                               metrics + R-hat/accept/swap diagnostics
  fig_dtf_distribution.png     distance-to-frontier across the optimized tracts
  fig_metric_shift.png         real vs best-counterfactual, per metric (with
                               the design-doc recommended bound drawn in)
  fig_best_networks.png        real (grey) vs MCMC-optimal (blue) street network
                               for the N tracts with the largest improvement

Usage: python viz_mcmc_spec.py [--examples 8]
"""
from __future__ import annotations
import argparse, json, pickle

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from uoi_common import OUT_DIR, ROOT

SAMP = OUT_DIR / "sampler_spec"
RES = ROOT / "results" / "mcmc_spec"
RES.mkdir(parents=True, exist_ok=True)

METRICS = ["link_node_ratio", "connected_node_ratio", "intersection_density",
           "median_block_length_ft", "walking_circuity", "pedshed_reach"]
LABEL = ["link-node ratio", "connected-node ratio", "intersection density",
         "median block length (ft)", "walking circuity", "pedshed reach"]
# design-doc recommended bound + whether higher is better (None = banded)
BOUND = [1.4, 0.7, 140, 600, (1.2, 1.7), None]
HIGHER = [True, True, True, False, None, True]


def load_summary() -> pd.DataFrame:
    s = json.loads((SAMP / "summary.json").read_text())
    rows = []
    for gid, d in s.items():
        row = {"GEOID": gid, "state": gid[:2],
               "distance_to_frontier": d["distance_to_frontier"],
               "best_E": d["best_E"], "swap_rate": d["swap_rate"],
               "accept_rate_cold": d["accept_rate_cold"],
               "rhat_max": max(d["rhat"].values()) if d["rhat"] else np.nan,
               "frontier_size": d["frontier_size"]}
        for i, m in enumerate(METRICS):
            row[m + "_real"] = d["u_real"][i]
            row[m + "_best"] = d["best_uoi"][i]
        rows.append(row)
    return pd.DataFrame(rows)


def fig_dtf(df):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    ax.hist(df["distance_to_frontier"], bins=40, color="#4C72B0")
    ax.axvline(df["distance_to_frontier"].median(), color="crimson", lw=2,
               label=f"median = {df['distance_to_frontier'].median():.3f}")
    ax.set_xlabel("distance to frontier (relative hypervolume shortfall)")
    ax.set_ylabel("number of tracts")
    ax.set_title(f"How far the top-{len(df)} real networks sit below their "
                 f"MCMC-achievable UOI frontier")
    ax.legend(); fig.tight_layout()
    fig.savefig(RES / "fig_dtf_distribution.png", dpi=140); plt.close(fig)
    print("saved fig_dtf_distribution.png")


def fig_metric_shift(df):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), facecolor="white")
    for ax, m, lab, b, hi in zip(axes.ravel(), METRICS, LABEL, BOUND, HIGHER):
        r, bst = df[m + "_real"], df[m + "_best"]
        ax.scatter(r, bst, s=8, alpha=0.4, color="#4C72B0")
        lim = [min(r.min(), bst.min()), max(r.max(), bst.max())]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6)  # y=x: no change
        if isinstance(b, tuple):
            for v in b:
                ax.axhline(v, color="green", lw=0.8, ls=":")
            ax.axvline(b[0], color="green", lw=0.6, ls=":", alpha=0.4)
        elif b is not None:
            ax.axhline(b, color="green", lw=0.9, ls=":",
                       label=f"rec {'≥' if hi else '≤'}{b}")
            ax.legend(fontsize=7)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("real"); ax.set_ylabel("MCMC-optimal")
        if m in ("intersection_density", "median_block_length_ft"):
            ax.set_xlim(lim); ax.set_ylim(lim)
    fig.suptitle("Real vs MCMC-optimal per metric (above y=x ⇒ metric raised; "
                 "green = design-doc bound)", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RES / "fig_metric_shift.png", dpi=140); plt.close(fig)
    print("saved fig_metric_shift.png")


def _draw(ax, G, pos, color, lw):
    segs = [[pos[u], pos[v]] for u, v in G.edges]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw))
    ax.set_aspect("equal"); ax.set_axis_off(); ax.autoscale()


def fig_best_networks(df, n_examples):
    sub = df.sort_values("distance_to_frontier", ascending=False).head(n_examples)
    cols = 4
    rows = (len(sub) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.1),
                             facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, r) in zip(axes, sub.iterrows()):
        p = SAMP / f"{r.GEOID}_w0_r0.pkl"
        if not p.exists():
            cand = sorted(SAMP.glob(f"{r.GEOID}_w*_r*.pkl"))
            p = cand[0] if cand else None
        if p is None:
            ax.set_axis_off(); continue
        d = pickle.load(open(p, "rb"))
        _draw(ax, d["G_real"], d["pos_real"], "#bbbbbb", 0.7)
        _draw(ax, d["best_G"], d["best_pos"], "#1f5fbf", 0.7)
        ax.set_title(f"{r.GEOID}  dtf={r.distance_to_frontier:.3f}\n"
                     f"circ {r.walking_circuity_real:.2f}→{r.walking_circuity_best:.2f}  "
                     f"ped {r.pedshed_reach_real:.3f}→{r.pedshed_reach_best:.3f}",
                     fontsize=7.5)
    for ax in axes[len(sub):]:
        ax.set_axis_off()
    fig.suptitle("MCMC-optimal network (blue) over the real network (grey) — "
                 "largest-improvement tracts", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RES / "fig_best_networks.png", dpi=145); plt.close(fig)
    print(f"saved fig_best_networks.png ({len(sub)} tracts)")


def _load_pkl(geoid):
    p = SAMP / f"{geoid}_w0_r0.pkl"
    if not p.exists():
        cand = sorted(SAMP.glob(f"{geoid}_w*_r*.pkl"))
        p = cand[0] if cand else None
    return pickle.load(open(p, "rb")) if p else None


def fig_optimal_gallery(df, n):
    """Grid of the MCMC-optimal networks (blue) over the faint real network
    (grey), for the n largest-improvement tracts — the 'organized optimal
    graphs' overview."""
    sub = df.sort_values("distance_to_frontier", ascending=False).head(n)
    cols = 6
    rows = (len(sub) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.7),
                             facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    drawn = 0
    for ax, (_, r) in zip(axes, sub.iterrows()):
        d = _load_pkl(r.GEOID)
        if d is None:
            ax.set_axis_off(); continue
        _draw(ax, d["G_real"], d["pos_real"], "#cfcfcf", 0.5)
        _draw(ax, d["best_G"], d["best_pos"], "#1f5fbf", 0.7)
        ax.set_title(f"{r.GEOID} ({r.state})\ndtf={r.distance_to_frontier:.3f}  "
                     f"E={r.best_E:.2f}", fontsize=7)
        drawn += 1
    for ax in axes[len(sub):]:
        ax.set_axis_off()
    fig.suptitle(f"MCMC-optimal street networks (blue) over real (grey) — "
                 f"top {n} by improvement", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(RES / "fig_optimal_gallery.png", dpi=150); plt.close(fig)
    print(f"saved fig_optimal_gallery.png ({drawn} networks)")


def export_individual(df, n):
    """One clean before→after PNG per tract for the top-n, under results/
    mcmc_spec/networks/ — each optimal graph as its own organized image."""
    out = RES / "networks"; out.mkdir(exist_ok=True)
    sub = df.sort_values("distance_to_frontier", ascending=False).head(n)
    for rank, (_, r) in enumerate(sub.iterrows(), 1):
        d = _load_pkl(r.GEOID)
        if d is None:
            continue
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 4.4), facecolor="white")
        _draw(a1, d["G_real"], d["pos_real"], "#444444", 0.8)
        a1.set_title("real", fontsize=10)
        _draw(a2, d["best_G"], d["best_pos"], "#1f5fbf", 0.9)
        a2.set_title("MCMC-optimal", fontsize=10)
        fig.suptitle(
            f"#{rank}  {r.GEOID} ({r.state})   dtf={r.distance_to_frontier:.3f}\n"
            f"link-node {r.link_node_ratio_real:.2f}→{r.link_node_ratio_best:.2f}   "
            f"circuity {r.walking_circuity_real:.2f}→{r.walking_circuity_best:.2f}   "
            f"pedshed {r.pedshed_reach_real:.3f}→{r.pedshed_reach_best:.3f}   "
            f"block(ft) {r.median_block_length_ft_real:.0f}→{r.median_block_length_ft_best:.0f}",
            fontsize=9, y=1.02)
        fig.tight_layout()
        fig.savefig(out / f"{rank:03d}_{r.GEOID}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"saved {len(sub)} individual before/after PNGs -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--gallery", type=int, default=24,
                    help="tracts in the optimal-network gallery grid")
    ap.add_argument("--export", type=int, default=12,
                    help="per-tract before/after PNGs to export individually")
    args = ap.parse_args()
    df = load_summary().sort_values("distance_to_frontier", ascending=False)
    df.round(4).to_csv(RES / "dtf_table.csv", index=False)
    print(f"dtf_table.csv ({len(df)} tracts) -> {RES}")
    fig_dtf(df)
    fig_metric_shift(df)
    fig_best_networks(df, args.examples)
    fig_optimal_gallery(df, args.gallery)
    export_individual(df, args.export)
    print("done.")


if __name__ == "__main__":
    main()
