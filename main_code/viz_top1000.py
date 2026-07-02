"""Composite UOI score and top-1000 ranking from the 6 spec metrics.

UOI_score = mean of national percentile ranks of the 6 metrics (block length
and circuity inverted); ties broken by n_ok. Writes ranked tables and figures
to results/top1000/.

Usage: python viz_top1000.py [--n 1000]
"""
from __future__ import annotations
import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from matplotlib.collections import LineCollection

from uoi_common import DATA, OUT_DIR, GRAPH_DIR, ROOT

RESULTS = ROOT / "results" / "top1000"
RESULTS.mkdir(parents=True, exist_ok=True)

METRICS = ["link_node_ratio", "connected_node_ratio", "intersection_density",
           "median_block_length_ft", "walking_circuity", "pedshed_reach"]
HIGHER_BETTER = {"link_node_ratio": True, "connected_node_ratio": True,
                 "intersection_density": True, "median_block_length_ft": False,
                 "walking_circuity": False, "pedshed_reach": True}
OK_FLAGS = ["lnr_ok", "cnr_ok", "inter_density_ok", "block_ok", "circuity_ok"]
LABEL = {"link_node_ratio": "link-node ratio", "connected_node_ratio": "connected-node ratio",
         "intersection_density": "intersection density", "median_block_length_ft": "median block length (ft)",
         "walking_circuity": "walking circuity", "pedshed_reach": "pedshed reach"}
STATE_NAME = {  # FIPS2 -> USPS
    "06": "CA", "36": "NY", "11": "DC", "42": "PA", "12": "FL", "25": "MA", "17": "IL",
    "24": "MD", "34": "NJ", "26": "MI", "53": "WA", "04": "AZ", "08": "CO", "39": "OH",
    "13": "GA", "37": "NC", "51": "VA", "72": "PR", "18": "IN", "29": "MO", "27": "MN",
    "55": "WI", "47": "TN", "09": "CT", "41": "OR", "01": "AL", "22": "LA", "21": "KY",
    "45": "SC", "40": "OK", "49": "UT", "35": "NM", "32": "NV", "33": "NH", "44": "RI",
    "10": "DE", "23": "ME", "15": "HI", "16": "ID", "19": "IA", "20": "KS", "28": "MS",
    "30": "MT", "31": "NE", "38": "ND", "46": "SD", "50": "VT", "54": "WV", "56": "WY",
    "02": "AK", "05": "AR"}


def build_scores() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "uoi_spec_metrics.csv", dtype={"GEOID": str})
    df = df[df["status"] == "ok"].dropna(subset=METRICS).copy()
    df["GEOID"] = df["GEOID"].str.zfill(11)
    df["state"] = df["GEOID"].str[:2]
    for c in METRICS:
        pct = df[c].rank(pct=True)
        df[c + "_pct"] = pct if HIGHER_BETTER[c] else (1 - pct)
    df["UOI_score"] = df[[c + "_pct" for c in METRICS]].mean(axis=1)
    df["n_ok"] = df[OK_FLAGS].sum(axis=1).astype(int)
    df = df.sort_values(["UOI_score", "n_ok"], ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def write_data(df: pd.DataFrame, top: pd.DataFrame) -> None:
    cols = (["rank", "GEOID", "state", "UOI_score", "n_ok", "n_nodes"]
            + METRICS + OK_FLAGS)
    top[cols].round(4).to_csv(RESULTS / "top1000_uoi.csv", index=False)
    top[cols].to_parquet(RESULTS / "top1000_uoi.parquet", index=False)
    df[["rank", "GEOID", "state", "UOI_score", "n_ok"] + METRICS].round(4).to_csv(
        RESULTS / "uoi_scores_all.csv", index=False)
    print(f"data -> {RESULTS}/top1000_uoi.csv  ({len(top)} rows of {len(df)} scored)")


def fig_score_distribution(df, top):
    cutoff = top["UOI_score"].min()
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    ax.hist(df["UOI_score"], bins=80, color="#4C72B0", alpha=0.85)
    ax.axvline(cutoff, color="crimson", lw=2,
               label=f"top-{len(top)} cutoff = {cutoff:.3f}")
    ax.set_xlabel("composite UOI score (mean of 6 metric percentiles)")
    ax.set_ylabel("number of census tracts")
    ax.set_title(f"National UOI score distribution — {len(df):,} scored tracts")
    ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "fig_score_distribution.png", dpi=140)
    plt.close(fig); print("saved fig_score_distribution.png")


def fig_metric_correlation(df):
    corr = df[METRICS].corr()
    n = len(METRICS)
    fig, ax = plt.subplots(figsize=(7.6, 6.6), facecolor="white")
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([LABEL[c] for c in METRICS], rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels([LABEL[c] for c in METRICS], fontsize=8)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else "black")
    ax.set_title("UOI metric inter-correlation (Pearson, national)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_metric_correlation.png", dpi=140)
    plt.close(fig); print("saved fig_metric_correlation.png")


def fig_by_state(top):
    vc = top["state"].map(lambda s: STATE_NAME.get(s, s)).value_counts().head(20)
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="white")
    ax.bar(vc.index, vc.values, color="#55A868")
    ax.set_ylabel(f"tracts in top {len(top)}")
    ax.set_title(f"Where the top {len(top)} UOI tracts are (top 20 states)")
    for i, v in enumerate(vc.values):
        ax.text(i, v + 1, str(v), ha="center", fontsize=8)
    plt.xticks(rotation=0); fig.tight_layout()
    fig.savefig(RESULTS / "fig_top1000_by_state.png", dpi=140)
    plt.close(fig); print("saved fig_top1000_by_state.png")


def fig_metric_profile(df, top):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), facecolor="white")
    for ax, c in zip(axes.ravel(), METRICS):
        ax.hist(df[c], bins=60, density=True, color="0.75", label="national")
        ax.hist(top[c], bins=40, density=True, color="crimson", alpha=0.6,
                label=f"top {len(top)}")
        ax.set_title(LABEL[c], fontsize=10)
        ax.set_yticks([])
        if c in ("intersection_density", "median_block_length_ft"):
            ax.set_xlim(df[c].quantile(0.005), df[c].quantile(0.97))
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"Metric profile: top {len(top)} (red) vs all scored tracts (grey)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RESULTS / "fig_metric_profile.png", dpi=140)
    plt.close(fig); print("saved fig_metric_profile.png")


def fig_networks(sub, fname, title):
    """Street networks of the tracts in `sub` as a 6-wide grid."""
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
        ax.add_collection(LineCollection(segs, colors="#222", linewidths=0.5))
        ax.set_title(f"#{int(r['rank'])} {r.GEOID}\n{STATE_NAME.get(r.state, r.state)}"
                     f"  UOI {r.UOI_score:.3f}", fontsize=7.5)
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
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    df = build_scores()
    top = df.head(args.n).copy()
    write_data(df, top)
    fig_score_distribution(df, top)
    fig_metric_correlation(df)
    fig_by_state(top)
    fig_metric_profile(df, top)
    # network grids: top, middle, and bottom 24 of the top-N
    N = len(top)
    mid = N // 2
    fig_networks(top.iloc[:24],
                 "fig_top24_networks.png",
                 f"Street networks — ranks 1-24 of the top {N} UOI tracts")
    fig_networks(top.iloc[mid - 12:mid + 12],
                 "fig_mid24_networks.png",
                 f"Street networks — ranks {mid-11}-{mid+12} (middle) of the top {N} UOI tracts")
    fig_networks(top.iloc[-24:],
                 "fig_last24_networks.png",
                 f"Street networks — ranks {N-23}-{N} (bottom) of the top {N} UOI tracts")
    print("done.")


if __name__ == "__main__":
    main()
