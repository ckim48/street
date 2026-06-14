"""Stage 1 (Geofabrik/pbf): per-tract pedestrian networks from state .pbf
extracts — the robust full-scale alternative to the Overpass-based
01_extract_networks.py.

Download one Geofabrik extract per state, then process it COUNTY BY COUNTY:
for each county, parse only that county's bbox out of the .pbf with pyrosm,
build + simplify the walk graph, and split it into per-tract subgraphs. Going
county-by-county (instead of whole-state) bounds peak memory — a whole-state
walk graph for a large state is ~7M+ nodes and OOMs a 62 GB box.

Usage:
    python 01_extract_networks_pbf.py --state 11          # DC (small smoke test)
    python 01_extract_networks_pbf.py --state 01          # Alabama
    python 01_extract_networks_pbf.py --state 06 --limit 20

Resumable: tracts with an existing GraphML are skipped, and a county whose
tracts are all done is never rebuilt. Failures are recorded in
data/outputs/extract_log.csv.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import requests
from pyrosm import OSM

from uoi_common import DATA, OUT_DIR, graph_path, tiger_tracts

PBF_DIR = DATA / "pbf"
PBF_DIR.mkdir(parents=True, exist_ok=True)

GEOFABRIK = "https://download.geofabrik.de/north-america/us/{slug}-latest.osm.pbf"
COUNTY_BUFFER_M = 300  # buffer county polygon so boundary streets aren't clipped

# Census state FIPS -> Geofabrik US sub-region slug
FIPS_SLUG = {
    "01": "alabama", "02": "alaska", "04": "arizona", "05": "arkansas",
    "06": "california", "08": "colorado", "09": "connecticut",
    "10": "delaware", "11": "district-of-columbia", "12": "florida",
    "13": "georgia", "15": "hawaii", "16": "idaho", "17": "illinois",
    "18": "indiana", "19": "iowa", "20": "kansas", "21": "kentucky",
    "22": "louisiana", "23": "maine", "24": "maryland",
    "25": "massachusetts", "26": "michigan", "27": "minnesota",
    "28": "mississippi", "29": "missouri", "30": "montana",
    "31": "nebraska", "32": "nevada", "33": "new-hampshire",
    "34": "new-jersey", "35": "new-mexico", "36": "new-york",
    "37": "north-carolina", "38": "north-dakota", "39": "ohio",
    "40": "oklahoma", "41": "oregon", "42": "pennsylvania",
    "44": "rhode-island", "45": "south-carolina", "46": "south-dakota",
    "47": "tennessee", "48": "texas", "49": "utah", "50": "vermont",
    "51": "virginia", "53": "washington", "54": "west-virginia",
    "55": "wisconsin", "56": "wyoming", "72": "puerto-rico",
}


def download_pbf(state: str) -> Path:
    slug = FIPS_SLUG[state]
    path = PBF_DIR / f"{slug}.osm.pbf"
    if path.exists() and path.stat().st_size > 0:
        print(f"  pbf cached: {path.name} ({path.stat().st_size/1e6:.0f} MB)", flush=True)
        return path
    url = GEOFABRIK.format(slug=slug)
    print(f"  downloading {url}", flush=True)
    t0 = time.time()
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(path)
    print(f"  saved {path.name} ({path.stat().st_size/1e6:.0f} MB, {time.time()-t0:.0f}s)",
          flush=True)
    return path


def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("yes", "true", "1", "-1")


def build_graph_for_area(pbf_path: Path, poly):
    """osmnx-compatible, simplified walk graph for one county-sized polygon.

    pyrosm parses only `poly`'s bbox from the .pbf (bounded memory). Boolean
    OSM tags are normalized so the GraphML round-trips through Stage 2, and the
    graph is simplified to the intersection topology used by the SF pilot.
    Returns None if the area has no walkable network.
    """
    osm = OSM(str(pbf_path), bounding_box=poly)
    nodes, edges = osm.get_network(network_type="walking", nodes=True)
    if edges is None or len(edges) == 0:
        return None
    G = osm.to_graph(nodes, edges, graph_type="networkx", network_type="walking")
    for _, _, d in G.edges(data=True):
        for key in ("oneway", "reversed"):
            if key in d:
                d[key] = _as_bool(d[key])
    G = ox.simplify_graph(G)
    return G


def truncate_tract(G, nodes_gdf, sindex, poly):
    """Subgraph of G for one tract: nodes inside the polygon plus their direct
    neighbors (the truncate_by_edge rule, so boundary-crossing edges survive).
    Uses a county-wide spatial index built once, so per-tract cost scales with
    nodes near the tract rather than the whole graph.
    """
    cand = list(sindex.intersection(poly.bounds))
    if not cand:
        raise ValueError("no candidate nodes")
    sub = nodes_gdf.iloc[cand]
    inside = sub.index[sub.intersects(poly)]
    if len(inside) == 0:
        raise ValueError("no nodes inside polygon")
    keep = set(inside)
    succ, pred = G._succ, G._pred
    for n in inside:
        keep.update(succ[n])
        keep.update(pred[n])
    return G.subgraph(keep).copy()


def extract_state(state: str, year: int, limit, log_writer) -> None:
    tracts = tiger_tracts(state, year).sort_values("GEOID")
    if limit:
        tracts = tracts.iloc[:limit]

    # save tract attributes for downstream stages (once)
    gpkg = DATA / f"tracts_{state}.gpkg"
    if not gpkg.exists():
        tracts[["GEOID", "COUNTYFP", "ALAND", "AWATER", "geometry"]].to_file(
            gpkg, driver="GPKG")

    pbf = download_pbf(state)
    utm = tracts.estimate_utm_crs()
    counties = sorted(tracts["COUNTYFP"].unique())
    print(f"state {state}: {len(tracts)} tracts in {len(counties)} counties",
          flush=True)
    t_state = time.time()

    for county in counties:
        ct = tracts[tracts["COUNTYFP"] == county]
        todo = [t for t in ct.itertuples() if not graph_path(t.GEOID).exists()]
        if not todo:
            continue
        # buffered county polygon, used both to bound the pbf parse and clip
        poly = (gpd.GeoSeries([ct.union_all()], crs=4326)
                .to_crs(utm).buffer(COUNTY_BUFFER_M).to_crs(4326).iloc[0])
        t0 = time.time()
        try:
            G = build_graph_for_area(pbf, poly)
        except Exception as e:  # one bad county must not abort the state
            for t in todo:
                log_writer.writerow([t.GEOID, f"county_err: {e}", 0, 0])
            print(f"  county {state}{county}: ERROR {e}", flush=True)
            continue
        if G is None:
            for t in todo:
                log_writer.writerow([t.GEOID, "empty_county", 0, 0])
            continue

        nodes_gdf = ox.graph_to_gdfs(G, edges=False)[["geometry"]]
        sindex = nodes_gdf.sindex
        for t in todo:
            try:
                sub = truncate_tract(G, nodes_gdf, sindex, t.geometry)
            except ValueError:
                log_writer.writerow([t.GEOID, "empty", 0, 0])
                continue
            ox.save_graphml(sub, graph_path(t.GEOID))
            log_writer.writerow([t.GEOID, "ok", len(sub), sub.number_of_edges()])
        print(f"  county {state}{county}: {len(todo)} tracts, "
              f"{len(G):,} nodes ({time.time()-t0:.0f}s)", flush=True)

    print(f"state {state}: done ({time.time()-t_state:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="2-digit state FIPS, e.g. 06")
    ap.add_argument("--year", type=int, default=2024, help="TIGER vintage")
    ap.add_argument("--limit", type=int, default=None, help="max tracts (smoke test)")
    args = ap.parse_args()

    if args.state not in FIPS_SLUG:
        raise SystemExit(f"no Geofabrik slug for state FIPS {args.state}")

    log_file = OUT_DIR / "extract_log.csv"
    new_log = not log_file.exists()
    with open(log_file, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["GEOID", "status", "n_nodes", "n_edges"])
        extract_state(args.state, args.year, args.limit, w)
        f.flush()


if __name__ == "__main__":
    main()
