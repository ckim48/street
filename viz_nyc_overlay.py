#!/usr/bin/env python3
"""Overlay our extracted street-network graph (nodes + edges) on a real NYC
basemap. Left: whole Manhattan network on a light map. Right: a neighborhood
zoom over satellite imagery so the node/edge abstraction is visible against
the real streets.

Usage: python viz_nyc_overlay.py
Output: results/figures/nyc_network_overlay.png
"""
import glob
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely import wkt as shapely_wkt
import contextily as cx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/wnlab/CK_street")
COUNTY = "36061"   # Manhattan (New York County)

def load_edges_nodes(graphml_paths):
    edges, nodes = [], []
    for fp in graphml_paths:
        try:
            G = nx.read_graphml(fp)
        except Exception:
            continue
        xy = {}
        for n, d in G.nodes(data=True):
            try:
                x, y = float(d["x"]), float(d["y"])
            except (KeyError, ValueError):
                continue
            xy[n] = (x, y)
            nodes.append((x, y))
        for u, v, d in G.edges(data=True):
            g = d.get("geometry")
            if g:
                try:
                    edges.append(shapely_wkt.loads(g)); continue
                except Exception:
                    pass
            if u in xy and v in xy:
                edges.append(LineString([xy[u], xy[v]]))
    e = gpd.GeoDataFrame(geometry=edges, crs="EPSG:4326").to_crs(3857)
    n = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in nodes],
                         crs="EPSG:4326").to_crs(3857)
    return e, n

paths = sorted(glob.glob(str(ROOT / f"data/graphs/{COUNTY}*.graphml")))
print(f"loading {len(paths)} Manhattan tract graphs ...", flush=True)
edges, nodes = load_edges_nodes(paths)
print(f"  {len(edges)} edges, {len(nodes)} nodes", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(18, 11), facecolor="white")

# ---- (A) whole Manhattan over a clean street basemap ----
edges.plot(ax=ax[0], color="#0066ff", linewidth=0.35, alpha=0.8)
cx.add_basemap(ax[0], source=cx.providers.CartoDB.PositronNoLabels, attribution_size=6)
ax[0].set_title(f"Our extracted network — all Manhattan ({len(paths)} tracts, "
                f"{len(edges):,} edges)", fontsize=12)
ax[0].set_axis_off()

# ---- (B) neighborhood zoom over satellite, nodes visible ----
# tighter zoom box around Midtown so nodes/edges are large and legible
box = gpd.GeoSeries([Point(-73.986, 40.750), Point(-73.974, 40.759)],
                    crs=4326).to_crs(3857)
xmin, ymin = box.iloc[0].x, box.iloc[0].y
xmax, ymax = box.iloc[1].x, box.iloc[1].y
ax[1].set_xlim(xmin, xmax); ax[1].set_ylim(ymin, ymax)
cx.add_basemap(ax[1], source=cx.providers.Esri.WorldImagery,
               attribution_size=6, zoom=17)
# dim the satellite so the neon network pops
ax[1].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                              facecolor="black", alpha=0.55, zorder=2))
edges.plot(ax=ax[1], color="#faff00", linewidth=2.2, alpha=0.95, zorder=3)
nodes.plot(ax=ax[1], color="#ff1744", markersize=34, alpha=1.0,
           edgecolor="white", linewidth=0.6, zorder=4)
ax[1].set_xlim(xmin, xmax); ax[1].set_ylim(ymin, ymax)
ax[1].set_title("Zoom: Midtown — nodes (red) + edges (yellow) on dimmed satellite",
                fontsize=12)
ax[1].set_axis_off()

fig.suptitle("NYC street network: graph abstraction overlaid on the real map",
             fontsize=15, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = ROOT / "results/figures/nyc_network_overlay.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"saved {out}")
