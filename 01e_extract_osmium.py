"""Stage 1 for OOM states (CA/TX): split the state .pbf into small TILE .pbf
files with `osmium extract` (one fast C++ pass), then parse each tile whole with
pyrosm (NO bounding_box -> the fast path, like the IL mega run) and slice tracts.

Why this finally works:
  * pyrosm's no-bbox parse is fast (IL: 10.3M nodes, 311s). ANY bounding_box
    (polygon OR rectangle) makes pyrosm filter per-element and is pathologically
    slow on big regions (a 25-county central-TX band ran >3h).
  * A whole-state parse is fast but OOMs CA(~38M)/TX(~20M nodes) on 62GB.
  * osmium splits the .pbf by bbox in seconds (6-10s/region, multi-region in one
    pass). Each tile .pbf is small -> pyrosm parses it WHOLE (fast, no bbox) and
    its in-memory GDFs are small -> bounded RAM. Best of both.

Tiles are a tract-count-balanced quantile grid; each tile's bbox = its tracts'
bounds + pad so boundary tracts keep their neighbors. One subprocess per tile
(memory isolation). Resumable: tracts with an existing GraphML skip; tile .pbf
files are cached.

Usage:
    python 01e_extract_osmium.py --state 48 [--per-cell 800]   # orchestrator
    python 01e_extract_osmium.py --state 48 --tile 3 --worker   # one tile
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, math, subprocess, sys, time
from pathlib import Path

import geopandas as gpd
import numpy as np
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

TILE_DIR = DATA / "pbf" / "tiles"
TILE_DIR.mkdir(parents=True, exist_ok=True)
PAD_DEG = 0.02  # ~2 km bbox pad around each tile's tracts


def _load_tracts(state, year):
    tracts = tiger_tracts(state, year).sort_values("GEOID")
    gpkg = DATA / f"tracts_{state}.gpkg"
    if not gpkg.exists():
        tracts[["GEOID", "COUNTYFP", "ALAND", "AWATER", "geometry"]].to_file(
            gpkg, driver="GPKG")
    return tracts


def manifest_path(state):
    return OUT_DIR / f"tiles_{state}.json"


def build_tiles(tracts, state, per_cell):
    """Tract-count-balanced quantile grid -> list of {idx, bbox, pbf, geoids}."""
    cen = tracts.geometry.representative_point()
    x, y = cen.x.values, cen.y.values
    n = len(tracts)
    ncells = max(1, math.ceil(n / per_cell))
    nx = max(1, round(math.sqrt(ncells)))
    ny = max(1, math.ceil(ncells / nx))
    xq = np.quantile(x, np.linspace(0, 1, nx + 1))
    yq = np.quantile(y, np.linspace(0, 1, ny + 1))
    xi = np.clip(np.digitize(x, xq[1:-1]), 0, nx - 1)
    yi = np.clip(np.digitize(y, yq[1:-1]), 0, ny - 1)
    geoids = tracts["GEOID"].to_numpy()
    geom = tracts.geometry
    tiles, idx = [], 0
    for cx in range(nx):
        for cy in range(ny):
            mask = (xi == cx) & (yi == cy)
            if not mask.any():
                continue
            minx, miny, maxx, maxy = geom[mask].total_bounds
            tiles.append({
                "idx": idx,
                "bbox": [float(minx - PAD_DEG), float(miny - PAD_DEG),
                         float(maxx + PAD_DEG), float(maxy + PAD_DEG)],
                "pbf": str(TILE_DIR / f"{state}_{idx}.osm.pbf"),
                "geoids": geoids[mask].tolist(),
            })
            idx += 1
    return tiles


def run_osmium(state, tiles):
    """One osmium pass writing every tile .pbf (skips tiles already present)."""
    todo = [t for t in tiles if not Path(t["pbf"]).exists()]
    if not todo:
        print("  all tile .pbf cached", flush=True); return
    cfg = {"extracts": [{"output": Path(t["pbf"]).name, "bbox": t["bbox"]}
                        for t in todo],
           "directory": str(TILE_DIR)}
    cfg_path = OUT_DIR / f"osmium_{state}.json"
    cfg_path.write_text(json.dumps(cfg))
    src = download_pbf(state)
    print(f"  osmium: extracting {len(todo)} tiles from {src.name}...", flush=True)
    t0 = time.time()
    subprocess.check_call(["osmium", "extract", "-c", str(cfg_path), str(src),
                           "--overwrite"])
    print(f"  osmium done ({time.time()-t0:.0f}s)", flush=True)


def run_worker(state, year, tile_idx):
    tiles = json.loads(manifest_path(state).read_text())
    tile = next(t for t in tiles if t["idx"] == tile_idx)
    tracts = _load_tracts(state, year).set_index("GEOID")
    sel = tracts.loc[[g for g in tile["geoids"] if g in tracts.index]]
    utm = tracts.estimate_utm_crs()
    todo = [(g, sel.loc[g, "geometry"]) for g in sel.index
            if not graph_path(g).exists()]
    print(f"  tile {tile_idx}: {len(sel)} tracts, {len(todo)} to build", flush=True)
    if not todo:
        return
    pbf = Path(tile["pbf"])
    if not pbf.exists():
        raise SystemExit(f"tile pbf missing: {pbf}")
    t0 = time.time()
    osm = OSM(str(pbf))  # whole small tile -> fast no-bbox parse
    nodes, edges = osm.get_network(network_type="walking", nodes=True)
    print(f"  parsed tile {tile_idx}: {len(edges):,} edges, {len(nodes):,} nodes "
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
        geoms = [g for _, g in todo]
        bufs = (gpd.GeoSeries(geoms, crs=4326).to_crs(utm)
                .buffer(TRACT_BUFFER_M).to_crs(4326))
        for i, ((geoid, geom), poly) in enumerate(zip(todo, bufs), 1):
            try:
                G = build_tract_graph(osm, edges, nodes, e_sindex, node_pos, poly)
                if G is None:
                    w.writerow([geoid, "empty", 0, 0]); n_empty += 1; continue
                ng = ox.graph_to_gdfs(G, edges=False)[["geometry"]]
                sub = truncate_tract(G, ng, ng.sindex, geom)
                ox.save_graphml(sub, graph_path(geoid))
                w.writerow([geoid, "ok", len(sub), sub.number_of_edges()]); n_ok += 1
            except ValueError:
                w.writerow([geoid, "empty", 0, 0]); n_empty += 1
            except Exception as e:  # noqa: BLE001
                w.writerow([geoid, f"tract_err: {e}", 0, 0]); n_err += 1
            if i % 100 == 0 or i == len(todo):
                f.flush()
                print(f"    {i}/{len(todo)} ok={n_ok} empty={n_empty} err={n_err} "
                      f"({i/(time.time()-t_build):.1f}/s)", flush=True)
    print(f"  tile {tile_idx} done ok={n_ok} empty={n_empty} err={n_err} "
          f"({time.time()-t_build:.0f}s)", flush=True)


def run_orchestrator(state, year, per_cell):
    tracts = _load_tracts(state, year)
    tiles = build_tiles(tracts, state, per_cell)
    manifest_path(state).write_text(json.dumps(tiles))
    sizes = [len(t["geoids"]) for t in tiles]
    undone = sum(1 for g in tracts["GEOID"] if not graph_path(g).exists())
    print(f"state {state}: {len(tracts)} tracts ({undone} undone) -> {len(tiles)} "
          f"tiles (sizes {min(sizes)}-{max(sizes)})", flush=True)
    run_osmium(state, tiles)
    here = str(Path(__file__))
    for t in tiles:
        rem = [g for g in t["geoids"] if not graph_path(g).exists()]
        if not rem:
            continue
        print(f"=== tile {t['idx']}/{len(tiles)}: {len(t['geoids'])} tracts "
              f"({len(rem)} undone) ===", flush=True)
        subprocess.call([sys.executable, here, "--state", state, "--year",
                         str(year), "--worker", "--tile", str(t["idx"])])
    print(f"state {state}: all tiles processed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--per-cell", type=int, default=800, help="target tracts/tile")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--worker", action="store_true")
    args = ap.parse_args()
    if args.state not in FIPS_SLUG:
        raise SystemExit(f"no Geofabrik slug for state FIPS {args.state}")
    if args.worker:
        run_worker(args.state, args.year, args.tile)
    else:
        run_orchestrator(args.state, args.year, args.per_cell)


if __name__ == "__main__":
    main()
