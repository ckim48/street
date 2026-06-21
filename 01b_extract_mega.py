"""Stage 1 for MEGA states (CA/IL/TX): parse the state .pbf ONCE, then build
every per-tract walk graph from the in-memory GeoDataFrames.

Why: pyrosm re-reads the full state .pbf on every get_network() call
(~330s for CA's 1.3GB regardless of bbox), so the per-county loop in
01_extract_networks_pbf.py costs one full parse PER COUNTY — infeasible for
TX (254 counties). And whole-county `to_graph` OOMs 62GB on LA/Cook/Harris.

This script pays ONE parse per state (the OOM was always in to_graph, never in
get_network, so the node/edge GDFs fit RAM), indexes the edges once, and for
each tract slices the edges by its buffered bbox, builds only that small graph,
simplifies, truncates to the tract, and saves. Bounded per-tract memory.

Output GraphML + extract_log rows are identical in form to the county script,
so Stage 2 (02b_compute_uoi_parallel.py) consumes them transparently.

Usage: python 01b_extract_mega.py --state 06 [--year 2024] [--limit N]
Resumable: tracts with an existing GraphML are skipped.
"""
from __future__ import annotations
import argparse, csv, importlib.util, time
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from pyrosm import OSM

from uoi_common import DATA, OUT_DIR, graph_path, tiger_tracts

# reuse helpers from the (digit-prefixed) county script — single source of truth
_spec = importlib.util.spec_from_file_location(
    "extract_pbf", str(Path(__file__).with_name("01_extract_networks_pbf.py")))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_as_bool = _mod._as_bool
truncate_tract = _mod.truncate_tract
download_pbf = _mod.download_pbf
FIPS_SLUG = _mod.FIPS_SLUG
TRACT_BUFFER_M = 300  # buffer tract polygon so boundary streets/neighbors survive


def build_tract_graph(osm, edges, nodes, e_sindex, node_pos, tract_poly_buf):
    """Small osmnx-compatible walk graph for one tract, sliced from state GDFs."""
    cand = list(e_sindex.intersection(tract_poly_buf.bounds))
    if not cand:
        return None
    e_sub = edges.iloc[cand]
    e_sub = e_sub[e_sub.intersects(tract_poly_buf)]
    if len(e_sub) == 0:
        return None
    nid = pd.unique(pd.concat([e_sub["u"], e_sub["v"]], ignore_index=True))
    pos = node_pos.reindex(nid).dropna().astype(int).to_numpy()
    n_sub = nodes.iloc[pos]
    G = osm.to_graph(n_sub, e_sub, graph_type="networkx", network_type="walking")
    for _, _, d in G.edges(data=True):
        for key in ("oneway", "reversed"):
            if key in d:
                d[key] = _as_bool(d[key])
    G = ox.simplify_graph(G)
    return G


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.state not in FIPS_SLUG:
        raise SystemExit(f"no Geofabrik slug for state FIPS {args.state}")

    tracts = tiger_tracts(args.state, args.year).sort_values("GEOID")
    if args.limit:
        tracts = tracts.iloc[: args.limit]
    gpkg = DATA / f"tracts_{args.state}.gpkg"
    if not gpkg.exists():
        tracts[["GEOID", "COUNTYFP", "ALAND", "AWATER", "geometry"]].to_file(
            gpkg, driver="GPKG")
    utm = tracts.estimate_utm_crs()
    todo = [t for t in tracts.itertuples() if not graph_path(t.GEOID).exists()]
    print(f"state {args.state}: {len(tracts)} tracts, {len(todo)} to build", flush=True)
    if not todo:
        return

    pbf = download_pbf(args.state)
    print("  parsing whole-state walk network (one pass)...", flush=True)
    t0 = time.time()
    osm = OSM(str(pbf))
    nodes, edges = osm.get_network(network_type="walking", nodes=True)
    print(f"  parsed: {len(edges):,} edges, {len(nodes):,} nodes "
          f"({time.time()-t0:.0f}s)", flush=True)
    e_sindex = edges.sindex
    node_pos = pd.Series(range(len(nodes)), index=nodes["id"].to_numpy())

    log_file = OUT_DIR / "extract_log.csv"
    new_log = not log_file.exists()
    n_ok = n_empty = n_err = 0
    t_build = time.time()
    with open(log_file, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["GEOID", "status", "n_nodes", "n_edges"])
        # buffer all todo polygons once (vectorized), in UTM then back to 4326
        bufs = (gpd.GeoSeries([t.geometry for t in todo], crs=4326)
                .to_crs(utm).buffer(TRACT_BUFFER_M).to_crs(4326))
        for i, (t, poly) in enumerate(zip(todo, bufs), 1):
            try:
                G = build_tract_graph(osm, edges, nodes, e_sindex, node_pos, poly)
                if G is None:
                    w.writerow([t.GEOID, "empty", 0, 0]); n_empty += 1; continue
                ng = ox.graph_to_gdfs(G, edges=False)[["geometry"]]
                sub = truncate_tract(G, ng, ng.sindex, t.geometry)
                ox.save_graphml(sub, graph_path(t.GEOID))
                w.writerow([t.GEOID, "ok", len(sub), sub.number_of_edges()]); n_ok += 1
            except ValueError:
                w.writerow([t.GEOID, "empty", 0, 0]); n_empty += 1
            except Exception as e:  # noqa: BLE001
                w.writerow([t.GEOID, f"tract_err: {e}", 0, 0]); n_err += 1
            if i % 100 == 0 or i == len(todo):
                f.flush()
                print(f"  {i}/{len(todo)} ok={n_ok} empty={n_empty} err={n_err} "
                      f"({i/(time.time()-t_build):.1f}/s)", flush=True)
    print(f"state {args.state}: done ok={n_ok} empty={n_empty} err={n_err} "
          f"({time.time()-t_build:.0f}s build, {time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
