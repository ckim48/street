"""Compute UOI for one state's already-extracted tract graphs and render a
correlation heatmap + choropleth maps. A per-state showcase of the pipeline
output, written to results/state_<FIPS>/.

Usage: python viz_state.py --state 01
"""
from __future__ import annotations

import argparse
import importlib.util

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uoi_common import DATA, GRAPH_DIR, ROOT

# reuse Stage 2's per-tract metric function
_spec = importlib.util.spec_from_file_location("s2", str(ROOT / "02_compute_uoi.py"))
s2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s2)

UOI = ["uoi_connectivity", "uoi_efficiency", "uoi_accessibility", "uoi_equity"]
UOI_TITLE = {"uoi_connectivity": "connectivity (link-node ratio)",
             "uoi_efficiency": "efficiency (1 / circuity)",
             "uoi_accessibility": "accessibility (800 m reach)",
             "uoi_equity": "equity (1 - Gini)"}
CORR_COLS = UOI + ["circuity_avg", "dead_end_frac", "orientation_entropy",
                   "n_intersections", "n_nodes", "reach_gini"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    args = ap.parse_args()
    state = args.state

    out = ROOT / "results" / f"state_{state}"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    geoids = sorted(p.stem for p in GRAPH_DIR.glob(f"{state}*.graphml"))
    print(f"state {state}: {len(geoids)} tract graphs", flush=True)
    rng = np.random.default_rng(42)
    rows = []
    for i, g in enumerate(geoids, 1):
        try:
            rows.append(s2.tract_metrics(g, rng))
        except Exception as e:
            rows.append({"GEOID": g, "status": f"error: {e}"})
        if i % 100 == 0:
            print(f"  {i}/{len(geoids)}", flush=True)
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    df.to_csv(out / "tables" / f"uoi_metrics_{state}.csv", index=False)
    ok[CORR_COLS].describe().T.round(4).to_csv(out / "tables" / f"uoi_summary_{state}.csv")
    print(f"scored {len(ok)}/{len(df)} -> {out/'tables'}", flush=True)

    # correlation heatmap
    corr = ok[CORR_COLS].corr(); n = len(CORR_COLS)
    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor="white")
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CORR_COLS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CORR_COLS, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 else "black")
    ax.set_title(f"UOI x morphology correlation - state {state} ({len(ok)} tracts)",
                 fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "figures" / f"uoi_correlation_{state}.png", dpi=130)
    plt.close(fig)

    # choropleth maps
    gpkg = DATA / f"tracts_{state}.gpkg"
    tracts = gpd.read_file(gpkg)[["GEOID", "geometry"]]
    gdf = tracts.merge(ok[["GEOID"] + UOI], on="GEOID", how="inner")
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), facecolor="white")
    for ax, col in zip(axes.ravel(), UOI):
        gdf.plot(column=col, cmap="viridis", scheme="quantiles", k=5, legend=True,
                 ax=ax, edgecolor="white", linewidth=0.1,
                 legend_kwds={"fontsize": 7, "loc": "lower left"})
        ax.set_title(UOI_TITLE[col], fontsize=11); ax.set_axis_off()
    fig.suptitle(f"Urban Optionality Index - state {state} census tracts",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "figures" / f"uoi_maps_{state}.png", dpi=130)
    plt.close(fig)
    print(f"figures -> {out/'figures'}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
