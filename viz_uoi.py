"""Organize Stage 2 UOI results into results/: a correlation heatmap, SF
choropleth maps of the four UOI dimensions, and tidied result tables.

Usage: python viz_uoi.py
Inputs : data/outputs/uoi_metrics.parquet, data/tracts_06075.gpkg
Outputs: results/figures/uoi_correlation.png
         results/figures/uoi_maps.png
         results/tables/uoi_metrics.csv
         results/tables/uoi_summary.csv
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uoi_common import DATA, OUT_DIR, ROOT

RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
TAB_DIR = RESULTS / "tables"
for d in (FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

UOI = ["uoi_connectivity", "uoi_efficiency", "uoi_accessibility", "uoi_equity"]
UOI_LABEL = {
    "uoi_connectivity": "connectivity\n(link-node ratio)",
    "uoi_efficiency": "efficiency\n(1 / circuity)",
    "uoi_accessibility": "accessibility\n(800 m reach)",
    "uoi_equity": "equity\n(1 - Gini)",
}
# UOI dims + morphology features for the correlation view
CORR_COLS = UOI + [
    "circuity_avg", "dead_end_frac", "orientation_entropy",
    "n_intersections", "n_nodes", "reach_gini",
]


def load() -> pd.DataFrame:
    df = pd.read_parquet(OUT_DIR / "uoi_metrics.parquet")
    return df[df["status"] == "ok"].copy()


def write_tables(df: pd.DataFrame) -> None:
    df.to_csv(TAB_DIR / "uoi_metrics.csv", index=False)
    summary = df[CORR_COLS].describe().T.round(4)
    summary.to_csv(TAB_DIR / "uoi_summary.csv")
    print(f"tables -> {TAB_DIR}  ({len(df)} tracts)")


def fig_correlation(df: pd.DataFrame) -> None:
    corr = df[CORR_COLS].corr()
    n = len(CORR_COLS)
    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor="white")
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CORR_COLS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CORR_COLS, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(v) > 0.55 else "black")
    ax.set_title("UOI dimensions × morphology — Pearson correlation\n"
                 "SF County (06075), 242 tracts", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    fig.tight_layout()
    out = FIG_DIR / "uoi_correlation.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}")


def fig_maps(df: pd.DataFrame) -> None:
    gpkg = DATA / "tracts_06075.gpkg"
    tracts = gpd.read_file(gpkg)[["GEOID", "geometry"]]
    gdf = tracts.merge(df[["GEOID"] + UOI], on="GEOID", how="inner")
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    print(f"map join: {len(gdf)}/{len(tracts)} tracts have UOI scores")

    fig, axes = plt.subplots(2, 2, figsize=(13, 13), facecolor="white")
    for ax, col in zip(axes.ravel(), UOI):
        gdf.plot(column=col, cmap="viridis", scheme="quantiles", k=5,
                 legend=True, ax=ax, edgecolor="white", linewidth=0.15,
                 legend_kwds={"fontsize": 7, "loc": "lower left"})
        ax.set_title(UOI_LABEL[col].replace("\n", " — "), fontsize=11)
        ax.set_axis_off()
    fig.suptitle("Urban Optionality Index across San Francisco census tracts",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG_DIR / "uoi_maps.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    df = load()
    write_tables(df)
    fig_correlation(df)
    fig_maps(df)
    print("done.")


if __name__ == "__main__":
    main()
