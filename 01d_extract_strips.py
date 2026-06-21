"""Stage 1 for OOM states (CA/TX): LONGITUDE-BAND extraction with RECTANGULAR
bbox parses.

Lessons that shaped this:
  * 01b (whole-state mega) parses once and is fast (IL: 10.3M nodes, 311s, 47GB)
    but OOMs CA (~38M nodes) / TX (~20M) — too many nodes for 62GB.
  * 01c (county-batch) used a POLYGON bbox = the union of many counties. pyrosm
    does per-element point-in-polygon against that complex shape -> pathological
    (a 25-county central-TX batch ran >3h without finishing). And large scattered
    batches OOM, falling back to per-county parses that each re-read the full .pbf.

So: split the state into N vertical bands of ~equal TRACT COUNT (a decent proxy
for node count in metro-dominated states — IL ~3160 nodes/tract). Each band's
network is ~IL-sized and FITS RAM. Parse each band ONCE with a RECTANGULAR bbox
([minx,miny,maxx,maxy] -> fast bounds filter, no polygon test), then slice every
tract from the in-memory GDFs (01b-style). One band per subprocess so an OOM
can't poison the rest; an OOM'd band is split in half and retried.

Output GraphML + extract_log rows match the other stage-1 scripts. Resumable:
tracts with an existing GraphML skip.

Usage:
    python 01d_extract_strips.py --state 48 [--per-band 1500]     # orchestrator
    python 01d_extract_strips.py --state 48 --geoids G1,G2 --worker
"""
from __future__ import annotations
import argparse, csv, importlib.util, math, subprocess, sys, time
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from pyrosm import OSM

from uoi_common import DATA, OUT_DIR, graph_path, tiger_tracts

_spec = importlib.util.spec_from_file_location(
    "extract_mega", str(Path(__file__).with_name("01b_extract_mega.py")))
_mega = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mega)
build_tract_graph = _mega.build_tract_graph
truncate_tract = _mega.truncate_tract
download_pbf = _mega.download_pbf
FIPS_SLUG = _mega.FIPS_SLUG
TRACT_BUFFER_M = _mega.TRACT_BUFFER_M
BAND_PAD_DEG = 0.03  # ~3 km rectangle pad so band-edge tracts keep their neighbors


def _load_tracts(state, year):
    tracts = tiger_tracts(state, year).sort_values("GEOID")
    gpkg = DATA / f"tracts_{state}.gpkg"
    if not gpkg.exists():
        tracts[["GEOID", "COUNTYFP", "ALAND", "AWATER", "geometry"]].to_file(
            gpkg, driver="GPKG")
    return tracts


def run_worker(state, year, geoids):
    """Build every undone tract in `geoids` from one rectangular-bbox parse."""
    tracts = _load_tracts(state, year)
    sel = tracts[tracts["GEOID"].isin(geoids)]
    utm = tracts.estimate_utm_crs()
    todo = [t for t in sel.itertuples() if not graph_path(t.GEOID).exists()]
    print(f"  band {len(geoids)} geoids: {len(todo)} to build", flush=True)
    if not todo:
        return
    pbf = download_pbf(state)
    # RECTANGULAR bbox of the undone tracts (+pad) -> fast pyrosm bounds filter
    minx, miny, maxx, maxy = gpd.GeoSeries([t.geometry for t in todo],
                                           crs=4326).total_bounds
    rect = [minx - BAND_PAD_DEG, miny - BAND_PAD_DEG,
            maxx + BAND_PAD_DEG, maxy + BAND_PAD_DEG]
    t0 = time.time()
    osm = OSM(str(pbf), bounding_box=rect)
    nodes, edges = osm.get_network(network_type="walking", nodes=True)
    print(f"  parsed band: {len(edges):,} edges, {len(nodes):,} nodes "
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
    print(f"  band done ok={n_ok} empty={n_empty} err={n_err} "
          f"({time.time()-t_build:.0f}s)", flush=True)


def make_bands(tracts, per_band):
    """Vertical bands of ~per_band tracts each, by centroid longitude."""
    cx = tracts.geometry.representative_point().x.values
    order = tracts.assign(_cx=cx).sort_values("_cx")["GEOID"].tolist()
    n = max(1, math.ceil(len(order) / per_band))
    size = math.ceil(len(order) / n)
    return [order[i:i + size] for i in range(0, len(order), size)]


def run_band(state, year, geoids, here, depth=0):
    """Run one band in a subprocess; on OOM-kill (rc=-9) split in half & recurse."""
    todo = [g for g in geoids if not graph_path(g).exists()]
    if not todo:
        return True
    rc = subprocess.call(
        [sys.executable, here, "--state", state, "--year", str(year),
         "--worker", "--geoids", ",".join(todo)])
    if rc == 0:
        return True
    if len(todo) <= 40 or depth >= 6:
        print(f"!!! band of {len(todo)} geoids failed (rc={rc}) and is too small "
              f"to split further — leaving as gap", flush=True)
        return False
    mid = len(todo) // 2
    print(f"!!! band of {len(todo)} died (rc={rc}); splitting into "
          f"{mid}+{len(todo)-mid}", flush=True)
    a = run_band(state, year, todo[:mid], here, depth + 1)
    b = run_band(state, year, todo[mid:], here, depth + 1)
    return a and b


def run_orchestrator(state, year, per_band):
    tracts = _load_tracts(state, year)
    bands = make_bands(tracts, per_band)
    undone = sum(1 for g in tracts["GEOID"] if not graph_path(g).exists())
    print(f"state {state}: {len(tracts)} tracts ({undone} undone) -> "
          f"{len(bands)} bands (~{per_band}/band)", flush=True)
    here = str(Path(__file__))
    failed = 0
    for bi, geoids in enumerate(bands, 1):
        rem = [g for g in geoids if not graph_path(g).exists()]
        print(f"=== band {bi}/{len(bands)}: {len(geoids)} tracts "
              f"({len(rem)} undone) ===", flush=True)
        if not run_band(state, year, geoids, here):
            failed += 1
    print(f"state {state}: bands done, {failed} bands left gaps", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--per-band", type=int, default=1500, help="tracts per band")
    ap.add_argument("--geoids", default=None, help="comma GEOID list (worker)")
    ap.add_argument("--worker", action="store_true")
    args = ap.parse_args()
    if args.state not in FIPS_SLUG:
        raise SystemExit(f"no Geofabrik slug for state FIPS {args.state}")
    if args.worker:
        geoids = [g.strip() for g in args.geoids.split(",") if g.strip()]
        run_worker(args.state, args.year, geoids)
    else:
        run_orchestrator(args.state, args.year, args.per_band)


if __name__ == "__main__":
    main()
