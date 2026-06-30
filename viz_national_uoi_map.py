"""National choropleth of the Urban Optionality Index across every U.S. census
tract (~84k).  Two figures:

  fig_national_uoi_map.png      one map, tracts colored by composite UOI_score
                                (CONUS main panel + Alaska / Hawaii / Puerto Rico
                                inset panels), percentile color scale.
  fig_national_uoi_metrics.png  6 small-multiple CONUS maps, one per design-doc
                                metric, each colored by its national percentile
                                in the "good" direction (block length & circuity
                                inverted so brighter = better everywhere).

Geometry: data/tracts_{ST}.gpkg (one per state, EPSG:4326).  UOI_score and the 6
metrics come from data/outputs/tract_panel.parquet (already carries the composite
score for all 84,395 scored tracts).  Maps are drawn in EPSG:5070 (Albers Equal
Area, CONUS) for the main panel and EPSG:4326 lon/lat for the AK/HI/PR insets.

Usage: python viz_national_uoi_map.py
"""
from __future__ import annotations
import glob, re

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import geopandas as gpd

from uoi_common import DATA, OUT_DIR, ROOT

RES = ROOT / "results" / "figures"
RES.mkdir(parents=True, exist_ok=True)

METRICS = ["link_node_ratio", "connected_node_ratio", "intersection_density",
           "median_block_length_ft", "walking_circuity", "pedshed_reach"]
LABEL = {"link_node_ratio": "link-node ratio",
         "connected_node_ratio": "connected-node ratio",
         "intersection_density": "intersection density",
         "median_block_length_ft": "median block length",
         "walking_circuity": "walking circuity",
         "pedshed_reach": "pedshed reach"}
HIGHER_BETTER = {"link_node_ratio": True, "connected_node_ratio": True,
                 "intersection_density": True, "median_block_length_ft": False,
                 "walking_circuity": False, "pedshed_reach": True}

# state FIPS groupings for the insets / CONUS split
AK, HI, PR = "02", "15", "72"
NONCONUS = {AK, HI, PR}


def load_geometry() -> gpd.GeoDataFrame:
    """Concatenate the per-state tract polygons (only the 2-digit state files,
    skipping the helper extracts like tracts_01ALL / tracts_06075)."""
    files = []
    for f in glob.glob(str(DATA / "tracts_*.gpkg")):
        m = re.search(r"tracts_(\d{2})\.gpkg$", f)
        if m:
            files.append((m.group(1), f))
    parts = []
    for st, f in sorted(files):
        g = gpd.read_file(f, columns=["GEOID", "geometry"])
        g["state"] = st
        parts.append(g)
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    # light simplify for fast rendering of 84k polygons (display only)
    gdf["geometry"] = gdf.geometry.simplify(0.002, preserve_topology=True)
    return gdf


def load_scores() -> pd.DataFrame:
    p = pd.read_parquet(OUT_DIR / "tract_panel.parquet")
    keep = ["GEOID", "UOI_score"] + [c for c in METRICS if c in p.columns]
    p = p[keep].copy()
    # per-metric national percentile in the "good" direction (for the 6-panel fig)
    for c in METRICS:
        if c in p.columns:
            pct = p[c].rank(pct=True)
            p[c + "_pct"] = pct if HIGHER_BETTER[c] else (1 - pct)
    return p


def _strip(ax):
    ax.set_axis_off()


def main():
    print("loading geometry (54 state files) ...", flush=True)
    gdf = load_geometry()
    sc = load_scores()
    gdf = gdf.merge(sc, on="GEOID", how="left")
    n_have = gdf["UOI_score"].notna().sum()
    print(f"{len(gdf):,} tract polygons, {n_have:,} with a UOI score", flush=True)

    conus = gdf[~gdf.state.isin(NONCONUS)].to_crs(epsg=5070)
    g_ak = gdf[gdf.state == AK]
    g_hi = gdf[gdf.state == HI]
    g_pr = gdf[gdf.state == PR]

    # ---------------- Figure 1: composite UOI_score national map ----------
    cmap = "viridis"
    norm = Normalize(vmin=0, vmax=1)
    pkw = dict(column="UOI_score", cmap=cmap, norm=norm, linewidth=0,
               edgecolor="none", missing_kwds={"color": "0.85"})

    fig = plt.figure(figsize=(15, 9), facecolor="white")
    ax = fig.add_axes([0.0, 0.18, 1.0, 0.80]); _strip(ax)        # CONUS
    conus.plot(ax=ax, **pkw)
    ax.set_title("Urban Optionality Index — composite score by census tract "
                 f"({n_have:,} U.S. tracts)", fontsize=15)

    ax_ak = fig.add_axes([0.02, 0.04, 0.22, 0.20]); _strip(ax_ak)
    if len(g_ak):
        g_ak.to_crs(epsg=3338).plot(ax=ax_ak, **pkw)
    ax_ak.set_title("Alaska", fontsize=9)

    ax_hi = fig.add_axes([0.26, 0.04, 0.14, 0.18]); _strip(ax_hi)
    if len(g_hi):
        g_hi.plot(ax=ax_hi, **pkw); ax_hi.set_xlim(-160.5, -154.5)
    ax_hi.set_title("Hawaii", fontsize=9)

    ax_pr = fig.add_axes([0.42, 0.04, 0.16, 0.16]); _strip(ax_pr)
    if len(g_pr):
        g_pr.plot(ax=ax_pr, **pkw)
    ax_pr.set_title("Puerto Rico", fontsize=9)

    cax = fig.add_axes([0.66, 0.07, 0.28, 0.025])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_label("composite UOI score  (0 = lowest optionality, 1 = highest)",
                 fontsize=10)
    fig.savefig(RES / "fig_national_uoi_map.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved {RES/'fig_national_uoi_map.png'}")

    # ---------------- Figure 2: 6-panel per-metric CONUS maps -------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), facecolor="white")
    for ax, c in zip(axes.ravel(), METRICS):
        _strip(ax)
        col = c + "_pct"
        if col not in conus.columns:
            continue
        conus.plot(ax=ax, column=col, cmap="magma", norm=Normalize(0, 1),
                   linewidth=0, edgecolor="none",
                   missing_kwds={"color": "0.85"})
        arrow = "↑ better" if HIGHER_BETTER[c] else "↓ better (inverted)"
        ax.set_title(f"{LABEL[c]}  ({arrow})", fontsize=11)
    cax = fig.add_axes([0.30, 0.04, 0.40, 0.02])
    cb = fig.colorbar(cm.ScalarMappable(norm=Normalize(0, 1), cmap="magma"),
                      cax=cax, orientation="horizontal")
    cb.set_label("national percentile in the favourable direction "
                 "(brighter = better)", fontsize=10)
    fig.suptitle("UOI design-doc metrics across U.S. census tracts (CONUS)",
                 fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(RES / "fig_national_uoi_metrics.png", dpi=150)
    plt.close(fig)
    print(f"saved {RES/'fig_national_uoi_metrics.png'}")


if __name__ == "__main__":
    main()
