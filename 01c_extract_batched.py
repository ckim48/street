"""Stage 1 for OOM-prone states (CA/TX): REGIONAL-BATCH extraction.

The whole-state mega parse (01b) holds the entire state walk network in RAM —
fine for IL (~10M nodes, 47GB) but OOMs CA (~38M) and TX (~20M) on a 62GB box.
The per-county script (01) bounds memory but re-parses the full .pbf once PER
county (pyrosm reads the whole file even for a bbox), which is far too slow for
TX's 254 counties.

This script splits the state's counties into BATCHES sized to fit RAM (a cap on
tracts per batch), and processes each batch in its OWN subprocess: parse the
batch's combined bbox once, then slice every tract from the in-memory GDFs
(01b-style). Subprocess isolation means an OOM-kill on one oversized batch can't
take down the rest — the orchestrator catches the death, retries that batch's
counties one-at-a-time, and records any single county that still OOMs as a gap
to backfill with finer tiling.

Output GraphML + extract_log rows match the county/mega scripts, so Stage 2
consumes them transparently. Resumable: tracts with an existing GraphML skip.

Usage:
    python 01c_extract_batched.py --state 06 [--cap 1200]      # orchestrator
    python 01c_extract_batched.py --state 06 --counties 037,038 --worker
"""
from __future__ import annotations
import argparse, csv, importlib.util, subprocess, sys, time
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from pyrosm import OSM

from uoi_common import DATA, OUT_DIR, graph_path, tiger_tracts

# reuse helpers (boolean-tag fix, per-tract slice, truncate) from the mega/county scripts
_spec = importlib.util.spec_from_file_location(
    "extract_mega", str(Path(__file__).with_name("01b_extract_mega.py")))
_mega = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mega)
build_tract_graph = _mega.build_tract_graph
truncate_tract = _mega.truncate_tract
download_pbf = _mega.download_pbf
FIPS_SLUG = _mega.FIPS_SLUG
TRACT_BUFFER_M = _mega.TRACT_BUFFER_M


def _load_tracts(state, year):
    tracts = tiger_tracts(state, year).sort_values("GEOID")
    gpkg = DATA / f"tracts_{state}.gpkg"
    if not gpkg.exists():
        tracts[["GEOID", "COUNTYFP", "ALAND", "AWATER", "geometry"]].to_file(
            gpkg, driver="GPKG")
    return tracts


def run_worker(state, year, counties=None, geoids=None):
    """Build every undone tract for the given counties (or explicit GEOID list)
    from a single bbox parse."""
    tracts = _load_tracts(state, year)
    if geoids is not None:
        sel = tracts[tracts["GEOID"].isin(geoids)]
        label = f"{len(geoids)} geoids"
    else:
        sel = tracts[tracts["COUNTYFP"].isin(counties)]
        label = ",".join(counties)
    utm = tracts.estimate_utm_crs()
    todo = [t for t in sel.itertuples() if not graph_path(t.GEOID).exists()]
    print(f"  batch {label}: {len(sel)} tracts, {len(todo)} to build",
          flush=True)
    if not todo:
        return
    pbf = download_pbf(state)
    # parse only the region spanning the UNDONE tracts (buffered) — on a resume
    # this is far smaller/cheaper than the whole batch's county footprint
    todo_union = gpd.GeoSeries([t.geometry for t in todo], crs=4326).union_all()
    bbox = (gpd.GeoSeries([todo_union], crs=4326)
            .to_crs(utm).buffer(TRACT_BUFFER_M).to_crs(4326).iloc[0])
    t0 = time.time()
    osm = OSM(str(pbf), bounding_box=bbox)
    nodes, edges = osm.get_network(network_type="walking", nodes=True)
    print(f"  parsed batch: {len(edges):,} edges, {len(nodes):,} nodes "
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
                print(f"    {i}/{len(todo)} ok={n_ok} empty={n_empty} err={n_err} "
                      f"({i/(time.time()-t_build):.1f}/s)", flush=True)
    print(f"  batch done ok={n_ok} empty={n_empty} err={n_err} "
          f"({time.time()-t_build:.0f}s)", flush=True)


def pack_batches(tracts, cap):
    """Greedily pack whole counties into batches of <= cap tracts (a county
    larger than cap becomes its own batch)."""
    counts = tracts.groupby("COUNTYFP").size().sort_index()
    batches, cur, cur_n = [], [], 0
    for county, n in counts.items():
        if cur and cur_n + n > cap:
            batches.append(cur); cur, cur_n = [], 0
        cur.append(county); cur_n += n
    if cur:
        batches.append(cur)
    return batches


def _worker_cmd(state, year, here, extra):
    return [sys.executable, here, "--state", state, "--year", str(year),
            "--worker"] + extra


def tile_county(state, year, county, cap, here):
    """Split one oversized county into spatially-contiguous tract chunks (each
    <= cap//3 tracts) so every chunk's bbox parse fits RAM. Returns True if all
    chunks succeeded."""
    tracts = _load_tracts(state, year)
    ct = tracts[tracts["COUNTYFP"] == county].copy()
    cen = ct.geometry.representative_point()
    ct["_x"], ct["_y"] = cen.x.values, cen.y.values
    ct = ct.sort_values(["_x", "_y"])  # spatial sort -> contiguous chunks
    chunk = max(150, cap // 3)
    geoid_chunks = [ct["GEOID"].iloc[i:i + chunk].tolist()
                    for i in range(0, len(ct), chunk)]
    print(f"  tiling county {state}{county}: {len(ct)} tracts -> "
          f"{len(geoid_chunks)} chunks of <= {chunk}", flush=True)
    ok = True
    for ci, gids in enumerate(geoid_chunks, 1):
        rc = subprocess.call(_worker_cmd(state, year, here,
                                         ["--geoids", ",".join(gids)]))
        if rc != 0:
            print(f"  !!! tile chunk {ci} of {state}{county} failed (rc={rc})",
                  flush=True)
            ok = False
    return ok


def run_orchestrator(state, year, cap):
    tracts = _load_tracts(state, year)
    batches = pack_batches(tracts, cap)
    print(f"state {state}: {len(tracts)} tracts -> {len(batches)} batches "
          f"(cap {cap})", flush=True)
    here = str(Path(__file__))
    failed_counties = []
    for bi, counties in enumerate(batches, 1):
        n = int(tracts["COUNTYFP"].isin(counties).sum())
        print(f"=== batch {bi}/{len(batches)}: {len(counties)} counties, "
              f"{n} tracts -> {','.join(counties)} ===", flush=True)
        rc = subprocess.call(_worker_cmd(state, year, here,
                                         ["--counties", ",".join(counties)]))
        if rc != 0:
            print(f"!!! batch {bi} died (rc={rc}); retrying counties solo", flush=True)
            for c in counties:
                rc2 = subprocess.call(_worker_cmd(state, year, here,
                                                  ["--counties", c]))
                if rc2 != 0:
                    print(f"!!! county {state}{c} died solo (rc={rc2}); tiling",
                          flush=True)
                    if not tile_county(state, year, c, cap, here):
                        failed_counties.append(c)
    if failed_counties:
        print(f"state {state}: counties STILL incomplete after tiling: "
              f"{','.join(failed_counties)}", flush=True)
    else:
        print(f"state {state}: all batches complete", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--cap", type=int, default=1200, help="max tracts per batch")
    ap.add_argument("--counties", default=None, help="comma COUNTYFP list (worker)")
    ap.add_argument("--geoids", default=None, help="comma GEOID list (worker, tiling)")
    ap.add_argument("--worker", action="store_true", help="run one batch in-process")
    args = ap.parse_args()
    if args.state not in FIPS_SLUG:
        raise SystemExit(f"no Geofabrik slug for state FIPS {args.state}")
    if args.worker:
        if args.geoids:
            geoids = [g.strip() for g in args.geoids.split(",") if g.strip()]
            run_worker(args.state, args.year, geoids=geoids)
        else:
            counties = [c.strip() for c in args.counties.split(",") if c.strip()]
            run_worker(args.state, args.year, counties=counties)
    else:
        run_orchestrator(args.state, args.year, args.cap)


if __name__ == "__main__":
    main()
