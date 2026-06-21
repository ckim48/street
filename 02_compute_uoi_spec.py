"""Stage 2 (spec): UOI metrics exactly per the design-doc 'UOI Index' table.

Six metrics (all on the projected, undirected tract graph; bounds from the doc):
  1 link_node_ratio        m/n                                   (Higher, rec>=1.4)
  2 connected_node_ratio   intersections/(intersections+deadends)(Higher, rec>=0.7)
  3 intersection_density   intersections / mile^2 (tract ALAND)  (Higher, rec>140)
  4 median_block_length_ft median street-segment length, feet    (Lower,  rec<=600)
  5 walking_circuity       mean over OD node pairs of net/straight(Lower, rec1.2-1.7)
  6 pedshed_reach          reachable street length within 400 m  (Higher)
                           per h3 lattice point / circle area (pi*400^2)

Usage: python 02_compute_uoi_spec.py [--limit N] [--states 01,02]
Resumable: GEOIDs already in data/outputs/uoi_spec_metrics.parquet are skipped.
"""
from __future__ import annotations
import argparse, glob, math, time
from statistics import median

import geopandas as gpd
import h3
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer

from uoi_common import GRAPH_DIR, OUT_DIR, DATA

REACH_M = 400.0
DISK_AREA = math.pi * REACH_M ** 2
N_OD = 500
H3_RES = 9
MIN_NODES = 5
M2_PER_MILE2 = 2_589_988.110336
FT_PER_M = 3.280839895
OUT = OUT_DIR / "uoi_spec_metrics.parquet"


def _latlngpoly(geom):
    """largest polygon ring as h3.LatLngPoly (lat,lng order)."""
    g = max(geom.geoms, key=lambda p: p.area) if geom.geom_type == "MultiPolygon" else geom
    ring = [(lat, lng) for lng, lat in g.exterior.coords]
    return h3.LatLngPoly(ring), g.centroid


def tract_metrics(geoid, aland_m2, geom, rng):
    G = ox.load_graphml(GRAPH_DIR / f"{geoid}.graphml")
    row = {"GEOID": geoid, "n_nodes": len(G)}
    if len(G) < MIN_NODES:
        row["status"] = "too_small"; return row
    Gu = ox.convert.to_undirected(ox.project_graph(G))
    n, m = len(Gu), Gu.number_of_edges()
    px = nx.get_node_attributes(Gu, "x"); py = nx.get_node_attributes(Gu, "y")
    degs = dict(Gu.degree())
    n_inter = sum(1 for d in degs.values() if d >= 3)
    n_dead = sum(1 for d in degs.values() if d == 1)

    # 1 link-node ratio
    row["link_node_ratio"] = m / n
    # 2 connected node ratio
    denom = n_inter + n_dead
    row["connected_node_ratio"] = n_inter / denom if denom else np.nan
    # 3 intersection density per mile^2 (tract land area)
    row["intersection_density"] = (n_inter / (aland_m2 / M2_PER_MILE2)
                                   if aland_m2 and aland_m2 > 0 else np.nan)
    # 4 median block length (median edge length -> ft)
    elens = [d.get("length", math.hypot(px[u]-px[v], py[u]-py[v]))
             for u, v, d in Gu.edges(data=True) if u != v]
    row["median_block_length_ft"] = median(elens) * FT_PER_M if elens else np.nan
    # 5 walking circuity over OD pairs (sample sources, single-source dijkstra)
    nodes = list(Gu.nodes)
    n_src = min(len(nodes), max(1, N_OD // 10))
    per = max(1, N_OD // n_src)
    ratios = []
    for s in rng.choice(nodes, size=n_src, replace=False):
        dist = nx.single_source_dijkstra_path_length(Gu, s, weight="length")
        tgts = [t for t in dist if t != s]
        if not tgts:
            continue
        for t in rng.choice(tgts, size=min(per, len(tgts)), replace=False):
            straight = math.hypot(px[s]-px[t], py[s]-py[t])
            if straight > 1:
                ratios.append(dist[t] / straight)
    row["walking_circuity"] = float(np.mean(ratios)) if ratios else np.nan
    # 6 structural pedshed reach (h3 lattice -> nearest node -> 400 m ego-graph)
    crs = Gu.graph["crs"]
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    poly, centroid = _latlngpoly(geom)
    try:
        cells = h3.polygon_to_cells(poly, H3_RES)
    except Exception:
        cells = []
    pts = ([h3.cell_to_latlng(c) for c in cells] if cells
           else [(centroid.y, centroid.x)])
    nx_arr = np.array([px[k] for k in nodes]); ny_arr = np.array([py[k] for k in nodes])
    vals = []
    for lat, lng in pts:
        X, Y = tf.transform(lng, lat)
        nn = nodes[int(np.argmin((nx_arr - X) ** 2 + (ny_arr - Y) ** 2))]
        ego = nx.ego_graph(Gu, nn, radius=REACH_M, distance="length")
        rl = sum(d.get("length", 0.0) for _, _, d in ego.edges(data=True))
        vals.append(rl / DISK_AREA)
    row["pedshed_reach"] = float(np.mean(vals)) if vals else np.nan
    row["n_lattice"] = len(pts)

    # bounds flags (per doc)
    row["lnr_ok"] = row["link_node_ratio"] >= 1.4
    row["cnr_ok"] = (row["connected_node_ratio"] >= 0.7
                     if row["connected_node_ratio"] == row["connected_node_ratio"] else False)
    row["inter_density_ok"] = row["intersection_density"] > 140
    row["block_ok"] = row["median_block_length_ft"] <= 600
    row["circuity_ok"] = 1.2 <= (row["walking_circuity"] or 99) <= 1.7
    row["status"] = "ok"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--states", default=None, help="comma list e.g. 01,02")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # tract land area + geometry from stage-1 gpkgs
    gdfs = [gpd.read_file(p)[["GEOID", "ALAND", "geometry"]]
            for p in sorted(glob.glob(str(DATA / "tracts_*.gpkg")))]
    tg = pd.concat(gdfs, ignore_index=True).drop_duplicates("GEOID").set_index("GEOID")

    done = set()
    prior = None
    if OUT.exists():
        prior = pd.read_parquet(OUT); done = set(prior["GEOID"])
    geoids = sorted(p.stem for p in GRAPH_DIR.glob("*.graphml") if p.stem not in done)
    if args.states:
        pre = tuple(args.states.split(","))
        geoids = [g for g in geoids if g[:2] in pre]
    if args.limit:
        geoids = geoids[: args.limit]
    print(f"{len(done)} done, {len(geoids)} to score", flush=True)

    rng = np.random.default_rng(args.seed)
    rows, t0 = [], time.time()
    for i, gid in enumerate(geoids, 1):
        try:
            geom = tg.loc[gid, "geometry"]; aland = tg.loc[gid, "ALAND"]
            rows.append(tract_metrics(gid, float(aland), geom, rng))
        except Exception as e:
            rows.append({"GEOID": gid, "status": f"error: {e}"})
        if i % 50 == 0 or i == len(geoids):
            print(f"  {i}/{len(geoids)} ({i/(time.time()-t0):.1f}/s)", flush=True)
    if rows:
        df = pd.DataFrame(rows)
        if prior is not None:
            df = pd.concat([prior, df], ignore_index=True)
        df.to_parquet(OUT, index=False)
        df.to_csv(OUT_DIR / "uoi_spec_metrics.csv", index=False)
        ok = df[df["status"] == "ok"]
        cols = ["link_node_ratio", "connected_node_ratio", "intersection_density",
                "median_block_length_ft", "walking_circuity", "pedshed_reach"]
        print(f"\nsaved {len(df)} ({len(ok)} ok) -> {OUT}", flush=True)
        if len(ok):
            print(ok[cols].describe().round(3).to_string())


if __name__ == "__main__":
    main()
