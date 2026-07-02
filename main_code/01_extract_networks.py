"""Stage 1: per-tract pedestrian street networks from OpenStreetMap (Overpass).

One walk-network download per county, split into tract subgraphs with
truncate_graph_polygon. Resumable: existing GraphML tracts are skipped;
failures are logged to data/outputs/extract_log.csv.

Usage: python 01_extract_networks.py --state 06 [--county 075]
"""
from __future__ import annotations

import argparse
import csv
import time

import geopandas as gpd
import osmnx as ox

from uoi_common import CACHE_DIR, DATA, OUT_DIR, graph_path, tiger_tracts

ox.settings.use_cache = True
ox.settings.cache_folder = str(CACHE_DIR)
ox.settings.log_console = False

# buffer (meters) around each county polygon so edge streets are not clipped
COUNTY_BUFFER_M = 300


def extract_county(tracts: gpd.GeoDataFrame, state: str, county: str, log_writer) -> None:
    """Download one county's walk network and split it into tract graphs."""
    todo = [t for t in tracts.itertuples() if not graph_path(t.GEOID).exists()]
    if not todo:
        print(f"  county {state}{county}: all {len(tracts)} tracts already extracted")
        return

    # union of tract polygons, buffered in a projected CRS, back to lat-lon
    poly = (
        gpd.GeoSeries([tracts.union_all()], crs=4326)
        .to_crs(tracts.estimate_utm_crs())
        .buffer(COUNTY_BUFFER_M)
        .to_crs(4326)
        .iloc[0]
    )
    print(f"  county {state}{county}: downloading walk network "
          f"({len(todo)}/{len(tracts)} tracts to extract)")
    t0 = time.time()
    G = ox.graph_from_polygon(poly, network_type="walk", simplify=True,
                              retain_all=True, truncate_by_edge=True)
    print(f"  county graph: {len(G):,} nodes, {G.number_of_edges():,} edges "
          f"({time.time() - t0:.0f}s)")

    for t in todo:
        geoid = t.GEOID
        try:
            sub = ox.truncate.truncate_graph_polygon(G, t.geometry, truncate_by_edge=True)
        except ValueError:
            # no graph nodes inside this tract (water tracts, parks, etc.)
            log_writer.writerow([geoid, "empty", 0, 0])
            continue
        ox.save_graphml(sub, graph_path(geoid))
        log_writer.writerow([geoid, "ok", len(sub), sub.number_of_edges()])
    print(f"  county {state}{county}: done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="2-digit state FIPS, e.g. 06")
    ap.add_argument("--county", default=None, help="3-digit county FIPS, e.g. 075")
    ap.add_argument("--year", type=int, default=2024, help="TIGER vintage")
    args = ap.parse_args()

    tracts = tiger_tracts(args.state, args.year)
    if args.county:
        counties = [args.county]
    else:
        counties = sorted(tracts["COUNTYFP"].unique())

    # save tract attributes (area, geometry) for downstream stages
    sel = tracts[tracts["COUNTYFP"].isin(counties)]
    gpkg = DATA / f"tracts_{args.state}{args.county or 'ALL'}.gpkg"
    sel[["GEOID", "COUNTYFP", "ALAND", "AWATER", "geometry"]].to_file(gpkg, driver="GPKG")
    print(f"saved {len(sel)} tract boundaries -> {gpkg}")

    log_file = OUT_DIR / "extract_log.csv"
    new_log = not log_file.exists()
    with open(log_file, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["GEOID", "status", "n_nodes", "n_edges"])
        for county in counties:
            ct = tracts[tracts["COUNTYFP"] == county]
            extract_county(ct, args.state, county, w)
            f.flush()


if __name__ == "__main__":
    main()
