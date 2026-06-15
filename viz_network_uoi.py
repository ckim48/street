"""For a few counties: draw the actual street network (nodes + edges) next to
a UOI choropleth, and dump the per-tract UOI. Shows what the UOI numbers mean
on the ground. Usage: python viz_network_uoi.py
"""
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import geopandas as gpd, networkx as nx, numpy as np, osmnx as ox, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
from uoi_common import DATA, GRAPH_DIR, ROOT

OUT = ROOT / "results" / "state_01"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
COUNTIES = {  # FIPS3 -> label
    "115": "St. Clair Co. (sprawl)",
    "081": "Lee Co. / Auburn (college town)",
    "033": "Colbert Co. (grid)",
}
UOI = ["uoi_connectivity", "uoi_efficiency", "uoi_accessibility", "uoi_equity"]

metrics = pd.read_csv(OUT / "tables" / "uoi_metrics_01.csv")
metrics = metrics[metrics["status"] == "ok"].copy()
metrics["GEOID"] = metrics["GEOID"].astype(np.int64).astype(str).str.zfill(11)
tracts = gpd.read_file(DATA / "tracts_01.gpkg")[["GEOID", "geometry"]]
tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)

kept = []
for fips3, label in COUNTIES.items():
    geoids = sorted(GRAPH_DIR.glob(f"01{fips3}*.graphml"))
    G = nx.compose_all([ox.load_graphml(p) for p in geoids])
    Gp = ox.project_graph(G)
    px = nx.get_node_attributes(Gp, "x"); py = nx.get_node_attributes(Gp, "y")

    segs = []
    for u, v, d in Gp.edges(data=True):
        if "geometry" in d:
            xs, ys = d["geometry"].xy
            segs.append(np.column_stack([np.asarray(xs), np.asarray(ys)]))
        else:
            segs.append([(px[u], py[u]), (px[v], py[v])])

    sub = metrics[metrics["GEOID"].str[2:5] == fips3]
    sub.assign(county=label).to_csv  # noqa (kept for clarity)
    kept.append(sub.assign(county_fips=fips3, county=label))
    mu = sub[UOI].mean()

    g = tracts.merge(sub[["GEOID"] + UOI], on="GEOID", how="inner").to_crs(Gp.graph["crs"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.4), facecolor="white")
    ax1.add_collection(LineCollection(segs, colors="0.45", linewidths=0.4))
    nx_x = np.array(list(px.values())); nx_y = np.array(list(py.values()))
    ax1.scatter(nx_x, nx_y, s=1.5, c="crimson", zorder=3)
    ax1.set_title(f"street network — {len(px):,} nodes / {Gp.number_of_edges():,} edges",
                  fontsize=10)
    ax1.set_aspect("equal"); ax1.set_axis_off(); ax1.autoscale()

    g.plot(column="uoi_connectivity", cmap="viridis", scheme="quantiles", k=5,
           legend=True, ax=ax2, edgecolor="white", linewidth=0.3,
           legend_kwds={"fontsize": 8, "loc": "lower left", "title": "connectivity"})
    ax2.set_title(f"UOI by tract ({len(g)} tracts)", fontsize=10)
    ax2.set_aspect("equal"); ax2.set_axis_off()

    fig.suptitle(f"{label}  —  mean UOI:  connectivity {mu.uoi_connectivity:.2f} | "
                 f"efficiency {mu.uoi_efficiency:.2f} | accessibility "
                 f"{mu.uoi_accessibility:.0f} | equity {mu.uoi_equity:.2f}",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT / "figures" / f"network_uoi_01{fips3}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved {out}  ({len(px):,} nodes, {len(g)} tracts)", flush=True)

allc = pd.concat(kept)[["county_fips", "county", "GEOID", "n_nodes", "n_edges_dir"] + UOI]
csv = OUT / "tables" / "network_counties_uoi.csv"
allc.round(4).to_csv(csv, index=False)
print(f"saved {csv}  ({len(allc)} tracts across {len(COUNTIES)} counties)", flush=True)
print("done.", flush=True)
