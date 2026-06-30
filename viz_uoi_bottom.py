"""National BOTTOM-24 UOI tracts: the 24 lowest-Urban-Optionality tracts in the
ENTIRE scored national set (~84k tracts) — NOT the bottom of the top-1000.

Mirror of viz_top1000.py's top-24 figure, but for the opposite extreme.  Uses
the identical composite score (mean of the 6 design-doc metric percentiles, with
block-length and circuity inverted) so the ranking is consistent with the
top-1000 work, then renders the actual street networks of the 24 worst tracts.

Inputs : data/outputs/uoi_spec_metrics.csv, data/graphs/*.graphml
Outputs: results/top1000/
    bottom24_uoi.csv                 the 24 lowest-UOI tracts + metrics
    fig_bottom24_networks.png        their street networks (6x4 grid)
Usage: python viz_uoi_bottom.py [--n 24]
"""
from __future__ import annotations
import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import osmnx as ox
from matplotlib.collections import LineCollection

from uoi_common import GRAPH_DIR, OUT_DIR, ROOT
from viz_top1000 import build_scores, METRICS, STATE_NAME

RESULTS = ROOT / "results" / "top1000"
RESULTS.mkdir(parents=True, exist_ok=True)


def draw_networks(sub, fname, title):
    """Render the street networks of the tracts in `sub` as a 6-wide grid.
    `sub` carries a national `rank` (1 = highest UOI, len = lowest)."""
    n = len(sub)
    cols, rows = 6, (n + 5) // 6
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 2.7),
                             facecolor="white")
    drawn = 0
    for ax, (_, r) in zip(axes.ravel(), sub.iterrows()):
        p = GRAPH_DIR / f"{r.GEOID}.graphml"
        try:
            G = ox.project_graph(ox.load_graphml(p))
        except Exception:
            ax.set_axis_off(); continue
        px = nx.get_node_attributes(G, "x"); py = nx.get_node_attributes(G, "y")
        segs = []
        for u, v, d in G.edges(data=True):
            if "geometry" in d:
                xs, ys = d["geometry"].xy
                segs.append(np.column_stack([np.asarray(xs), np.asarray(ys)]))
            else:
                segs.append([(px[u], py[u]), (px[v], py[v])])
        ax.add_collection(LineCollection(segs, colors="#7a2020", linewidths=0.5))
        ax.set_title(f"#{int(r['rank'])}/{r['n_total']}  {r.GEOID}\n"
                     f"{STATE_NAME.get(r.state, r.state)}  UOI {r.UOI_score:.3f}",
                     fontsize=7.5)
        ax.set_aspect("equal"); ax.set_axis_off(); ax.autoscale()
        drawn += 1
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=13, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(RESULTS / fname, dpi=145)
    plt.close(fig); print(f"saved {fname} ({drawn} networks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    df = build_scores()                       # ranked desc by UOI_score (rank 1 = best)
    n_total = len(df)
    bottom = df.tail(args.n).iloc[::-1].copy()  # worst first (lowest UOI)
    bottom["n_total"] = n_total
    cols = ["rank", "GEOID", "state", "UOI_score", "n_ok", "n_nodes"] + METRICS
    bottom[cols].round(4).to_csv(RESULTS / "bottom24_uoi.csv", index=False)
    print(f"data -> {RESULTS}/bottom24_uoi.csv  "
          f"(national ranks {n_total-args.n+1}-{n_total} of {n_total})")
    draw_networks(
        bottom, "fig_bottom24_networks.png",
        f"Street networks — the {args.n} LOWEST-UOI tracts nationally "
        f"(ranks {n_total-args.n+1}-{n_total} of {n_total:,})")
    print("done.")


if __name__ == "__main__":
    main()
