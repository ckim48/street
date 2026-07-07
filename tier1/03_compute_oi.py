"""Tier 1 - Step 3.2: local Oi on both sides of each HOLC boundary, per decade.

This turns the boundary inventory + decade graphs into the RD running-variable
table.  For each boundary segment we walk out along the segment normal to a grid
of signed offsets (- = higher-grade side, + = lower-grade / treated side) and,
at each offset point, compute the six-metric Oi on the decade graph clipped to a
small disk.  The result is a long table keyed by (segment, decade, signed
distance) that Step 4 feeds straight into a geographic RD.

The signed-distance orientation is set per segment by testing which normal
direction lands inside the lower-grade region, so `treat = offset > 0` is always
"the worse-rated side" (Guide Step 5.2).

Outputs (data/tier1/oi/): {slug}_oi.parquet  (one row per sample point)
Usage:
  python tier1/03_compute_oi.py --cities chicago --barriers street
  python tier1/03_compute_oi.py --radius 250 --offsets 75,150,225,300 --max-seg 400
"""
from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

from oi_local import METRIC_NAMES, local_oi
from tier1_common import (BND_DIR, CITIES, DECADES, GRAPH_DIR, METRIC_CRS,
                          OI_DIR, city_slugs, clean_grade)

warnings.filterwarnings("ignore")


def load_decade_graphs(slug):
    """decade(int|'present') -> (Graph, cKDTree, node_ids, node_xy)."""
    out = {}
    paths = [(D, GRAPH_DIR / f"{slug}_{D}.graphml") for D in DECADES]
    paths.append(("present", GRAPH_DIR / f"{slug}_present.graphml"))
    for D, p in paths:
        if not p.exists():
            continue
        G = nx.read_graphml(p)
        # graphml stores attrs as str; coerce
        for _, d in G.nodes(data=True):
            d["x"] = float(d["x"]); d["y"] = float(d["y"])
        for _, _, d in G.edges(data=True):
            d["length"] = float(d.get("length", 0.0))
        node_ids = list(G.nodes)
        xy = np.array([[G.nodes[u]["x"], G.nodes[u]["y"]] for u in node_ids])
        out[D] = (G, cKDTree(xy), node_ids, xy)
    return out


def seg_normal(geom):
    """Unit normal at the segment midpoint (from its overall direction)."""
    cs = np.asarray(geom.coords)
    d = cs[-1] - cs[0]
    L = np.hypot(*d) or 1.0
    tx, ty = d / L
    return np.array([-ty, tx])                      # 90-deg rotation


def lower_side_sign(midpoint, normal, lo_region, eps=40.0):
    """+1 if midpoint+eps*normal is inside the lower-grade region, else -1."""
    p = Point(midpoint + eps * normal)
    if lo_region is not None and lo_region.contains(p):
        return 1.0
    return -1.0


def clip_oi(center, radius, G, tree, node_ids, rng):
    idx = tree.query_ball_point(center, radius)
    if len(idx) < 5:
        return None
    H = G.subgraph([node_ids[i] for i in idx])
    disk_area = np.pi * radius ** 2
    return local_oi(H, disk_area, rng)


def process_city(slug, offsets, radius, barriers, max_seg, decades):
    print(f"[{slug}] {CITIES[slug]['city']}")
    bpath = BND_DIR / f"{slug}_boundaries.gpkg"
    if not bpath.exists():
        print("  no boundary file; run 01 first"); return None
    segs = gpd.read_file(bpath).to_crs(METRIC_CRS)
    if barriers != "all":
        segs = segs[segs.barrier.isin(barriers.split(","))]
    graphs = load_decade_graphs(slug)
    if not graphs:
        print("  no decade graphs; run 02 first"); return None
    if decades:
        want = set(decades.split(","))
        graphs = {D: v for D, v in graphs.items() if str(D) in want}
    grades = gpd.read_file(BND_DIR / f"{slug}_grades.gpkg").to_crs(METRIC_CRS)
    grades["grade"] = grades["grade"].map(clean_grade)
    region = {g: geom for g, geom in zip(grades.grade, grades.geometry)}

    offs = [float(x) for x in offsets.split(",")]
    rng = np.random.default_rng(7)
    rows = []
    for pair, gsub in segs.groupby("pair"):
        lo = pair.split("-")[1]
        lo_region = region.get(lo)
        if max_seg and len(gsub) > max_seg:
            gsub = gsub.sample(max_seg, random_state=1)
        print(f"  {pair}: {len(gsub)} segments x {len(offs)*2} offsets x {len(graphs)} decades")
        for _, s in gsub.iterrows():
            mid = np.asarray(s.geometry.interpolate(0.5, normalized=True).coords[0])
            nrm = seg_normal(s.geometry)
            sign = lower_side_sign(mid, nrm, lo_region)
            for o in offs:
                for direction in (+1.0, -1.0):
                    center = tuple(mid + direction * o * nrm)   # geometric point
                    signed = direction * o * sign               # + => lower/treated
                    for D, (G, tree, nids, xy) in graphs.items():
                        oi = clip_oi(center, radius, G, tree, nids, rng)
                        if oi is None:
                            continue
                        rows.append(dict(
                            slug=slug, seg_id=s.seg_id, pair=pair,
                            hi_grade=s.hi_grade, lo_grade=s.lo_grade,
                            barrier=s.barrier, decade=str(D),
                            signed_dist=float(signed),
                            treat=int(signed > 0),
                            **{k: oi[k] for k in METRIC_NAMES},
                            n_clip_nodes=oi["n_nodes"]))
    if not rows:
        print("  no samples"); return None
    df = pd.DataFrame(rows)
    df.to_parquet(OI_DIR / f"{slug}_oi.parquet", index=False)
    print(f"  -> {OI_DIR/f'{slug}_oi.parquet'}  ({len(df):,} rows)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=city_slugs())
    ap.add_argument("--offsets", default="75,150,225,300",
                    help="unsigned perpendicular offsets in metres")
    ap.add_argument("--radius", type=float, default=250.0,
                    help="disk radius for the local Oi clip (m)")
    ap.add_argument("--barriers", default="street",
                    help="comma list of barrier classes to keep, or 'all'")
    ap.add_argument("--max-seg", type=int, default=400,
                    help="cap segments per (city,pair) (0 = no cap)")
    ap.add_argument("--decades", default="",
                    help="comma subset of decades e.g. 1940,1980,2020,present")
    args = ap.parse_args()
    for slug in args.cities:
        try:
            process_city(slug, args.offsets, args.radius, args.barriers,
                         args.max_seg, args.decades)
        except Exception as e:                                    # noqa: BLE001
            print(f"[{slug}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
