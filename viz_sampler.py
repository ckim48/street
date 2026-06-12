"""Render sampler results: real network vs best counterfactual per tract,
plus the posterior UOI cloud with the Pareto frontier and the real point.

Usage: python viz_sampler.py 06075012802 06075021600 06075030500
"""
from __future__ import annotations

import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uoi_common import OUT_DIR

SAMPLER_DIR = OUT_DIR / "sampler"
DIMS = ["connectivity", "efficiency", "accessibility", "equity"]


def draw_graph(ax, G, pos, title):
    for a, b in G.edges:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color="black", linewidth=0.7, zorder=1)
    xy = np.array(list(pos.values()))
    ax.scatter(xy[:, 0], xy[:, 1], s=4, color="crimson", zorder=2)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_axis_off()


def main(geoids):
    for geoid in geoids:
        chains = []
        for pkl in sorted(SAMPLER_DIR.glob(f"{geoid}_w*_r*.pkl")):
            with open(pkl, "rb") as f:
                chains.append(pickle.load(f))
        if not chains:
            print(f"{geoid}: no chain files")
            continue
        best = max((c for c in chains if c["best_G"] is not None),
                   key=lambda c: c["best_E"])
        u_real = np.array(chains[0]["u_real"])
        cloud = np.vstack([np.array(c["samples"]) for c in chains if c["samples"]])

        fig = plt.figure(figsize=(16, 5), facecolor="white")
        ax1 = fig.add_subplot(1, 3, 1)
        draw_graph(ax1, chains[0]["G_real"], chains[0]["pos_real"],
                   f"{geoid} REAL\nuoi={np.round(u_real, 3)}")
        ax2 = fig.add_subplot(1, 3, 2)
        draw_graph(ax2, best["best_G"], best["best_pos"],
                   f"BEST counterfactual (E={best['best_E']:.3f})\n"
                   f"uoi={np.round(np.array(best['best_uoi']), 3)}")

        # posterior cloud: accessibility vs equity, colored by connectivity
        ax3 = fig.add_subplot(1, 3, 3)
        sc = ax3.scatter(cloud[:, 2], cloud[:, 3], c=cloud[:, 0], s=8,
                         cmap="viridis", alpha=0.6)
        ax3.scatter([u_real[2]], [u_real[3]], marker="*", s=250, color="red",
                    edgecolor="black", zorder=5, label="real network")
        plt.colorbar(sc, ax=ax3, shrink=0.8, label="connectivity")
        ax3.set_xlabel("accessibility (reach length, m)")
        ax3.set_ylabel("equity (1 - Gini)")
        ax3.set_title("posterior UOI cloud (cold chains)", fontsize=10)
        ax3.legend(fontsize=9)

        fig.tight_layout()
        out = SAMPLER_DIR / f"{geoid}_counterfactual.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
