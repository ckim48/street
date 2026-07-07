"""Tier 1 - Step 3.1: per-city decade road graphs from CHRONEX-US.

CHRONEX-US ships one GeoPackage of dated road polylines per CBSA
(chronex_us_<cbsa>.gpkg, local UTM).  Each segment carries an estimated
construction-year band; we filter to segments that exist *by* each decade to
reconstruct the 1940/1950/.../2020 street network, clipped to the city's HOLC
footprint (+buffer) so the graphs stay small and boundary-focused.

Year encoding (CHRONEX data records): `yr_upper_M1` is the upper bound of the
estimated construction epoch, in calendar years; value 0 means "pre-1900".  A
segment is present by decade D if yr==0 or 0<yr<=D.  MTFCC_CODE S1100 (primary /
limited-access) is dropped by default so the graph reflects the local street
fabric the Oi index is about.

If a city's CHRONEX gpkg is absent, `--present-osm` builds a single present-day
OpenStreetMap snapshot instead ({slug}_present.graphml) so the rest of the
pipeline is runnable end-to-end before the 2.7 GB CHRONEX download lands.

Outputs (data/tier1/graphs/): {slug}_{decade}.graphml  (metric CRS, 'length' m)
Usage:
  python tier1/02_decade_graphs.py --cities chicago
  python tier1/02_decade_graphs.py --present-osm            # OSM fallback for all
"""
from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np

from oi_local import graph_from_lines
from tier1_common import (BND_DIR, CITIES, DECADES, GRAPH_DIR, HOLC_DIR,
                          METRIC_CRS, city_slugs)

warnings.filterwarnings("ignore")

CHRONEX_DIR = HOLC_DIR.parent / "chronex"
CLIP_BUFFER_M = 2000.0
DROP_MTFCC = {"S1100", "S1630"}          # interstates/primary + ramps
DEFAULT_YEAR_FIELD = "yr_upper_M1"


def _col(gdf, name):
    """Case-insensitive column lookup."""
    low = {c.lower(): c for c in gdf.columns}
    return low.get(name.lower())


def city_footprint(slug):
    gp = BND_DIR / f"{slug}_grades.gpkg"
    if not gp.exists():
        return None
    g = gpd.read_file(gp).to_crs(METRIC_CRS)
    return g.union_all().buffer(CLIP_BUFFER_M)


def build_from_chronex(slug, year_field):
    cbsa = CITIES[slug]["cbsa"]
    gpkg = CHRONEX_DIR / f"chronex_us_{cbsa}.gpkg"
    if not gpkg.exists():
        print(f"  [chronex] missing {gpkg.name}")
        return False
    g = gpd.read_file(gpkg).to_crs(METRIC_CRS)
    yf = _col(g, year_field) or _col(g, DEFAULT_YEAR_FIELD)
    mf = _col(g, "MTFCC_CODE")
    if yf is None:
        print(f"  [chronex] no year field '{year_field}' in {list(g.columns)}")
        return False
    fp = city_footprint(slug)
    if fp is not None:
        g = g[g.intersects(fp)].copy()
        g["geometry"] = g.geometry.intersection(fp)
    if mf is not None:
        g = g[~g[mf].astype(str).str.upper().isin(DROP_MTFCC)]
    yr = g[yf].fillna(-1).astype(float).values
    print(f"  [chronex] {len(g)} local segments in footprint "
          f"(year field {yf}; yr range {int(np.nanmin(yr[yr>0]) if (yr>0).any() else 0)}"
          f"-{int(np.nanmax(yr))})")
    for D in DECADES:
        present = (yr == 0) | ((yr > 0) & (yr <= D))
        sub = g.loc[present, "geometry"]
        H = graph_from_lines(sub.values)
        if H.number_of_nodes() == 0:
            print(f"    {D}: empty"); continue
        H.graph["crs"] = f"EPSG:{METRIC_CRS}"; H.graph["decade"] = D; H.graph["city"] = slug
        nx.write_graphml(H, GRAPH_DIR / f"{slug}_{D}.graphml")
        print(f"    {D}: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    return True


def build_present_osm(slug):
    import osmnx as ox
    fp_ll = None
    gp = BND_DIR / f"{slug}_grades.gpkg"
    if gp.exists():
        fp_ll = gpd.read_file(gp).to_crs(4326).union_all().convex_hull.buffer(0.02)
    if fp_ll is None:
        print("  [osm] no footprint; skip"); return False
    print("  [osm] downloading present-day drive network ...")
    G = ox.graph_from_polygon(fp_ll, network_type="drive", simplify=True)
    Gp = ox.project_graph(G, to_crs=f"EPSG:{METRIC_CRS}")
    H = nx.Graph()
    for n, d in Gp.nodes(data=True):
        H.add_node(n, x=float(d["x"]), y=float(d["y"]))
    for u, v, d in Gp.edges(data=True):
        L = float(d.get("length", 0.0))
        if not H.has_edge(u, v) or L < H.edges[u, v]["length"]:
            H.add_edge(u, v, length=L)
    H.graph["crs"] = f"EPSG:{METRIC_CRS}"; H.graph["decade"] = "present"
    nx.write_graphml(H, GRAPH_DIR / f"{slug}_present.graphml")
    print(f"  [osm] present: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=city_slugs())
    ap.add_argument("--year-field", default=DEFAULT_YEAR_FIELD)
    ap.add_argument("--present-osm", action="store_true",
                    help="also/instead build a present-day OSM snapshot per city")
    args = ap.parse_args()
    for slug in args.cities:
        print(f"[{slug}] {CITIES[slug]['city']}")
        ok = build_from_chronex(slug, args.year_field)
        if args.present_osm or not ok:
            build_present_osm(slug)


if __name__ == "__main__":
    main()
