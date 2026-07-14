"""PathFinder pre-highway network (Regime 1) — real historical streets.

The pre-highway network is fetched from **OpenHistoricalMap** (OHM), whose ways
carry `start_date` / `end_date` tags, so we can reconstruct the street grid as
it stood in construction_year-1 -- INCLUDING the streets the interstate and its
urban-renewal clearance later demolished (which the modern TIGER network no
longer contains).  Coverage is city-specific: Detroit's Black Bottom / Paradise
Valley is richly mapped in OHM (≈1000 dated streets at 1958, ≈240 later removed);
other cities are sparse (see pf_prehist_coverage()).

fetch_ohm_prehighway(slug) queries the OHM Overpass API over Omega's bbox, keeps
vehicular ways alive at build_start-1, reprojects to the city UTM, clips to Ω,
and caches to data/pathfinder/prehist/{slug}_pre.gpkg so the sampler is offline.
build_pre_graph(slug) returns (G, pos, layers) mirroring pf_graph.build_modern_graph.
"""
from __future__ import annotations

import subprocess

import geopandas as gpd
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from pf_common import BND_DIR, CITIES, PF
from pf_graph import _graph_from_lines, _noded_lines

import networkx as nx

PREHIST_DIR = PF / "prehist"
PREHIST_DIR.mkdir(parents=True, exist_ok=True)

OHM_OVERPASS = "https://overpass-api.openhistoricalmap.org/api/interpreter"

# non-vehicular OHM highway classes we drop (keep the street grid only)
_DROP_HWY = {"path", "footway", "steps", "cycleway", "bridleway",
             "construction", "proposed", "corridor", "platform"}


def _yr(s):
    if not s:
        return None
    try:
        return int(str(s)[:4])
    except (ValueError, TypeError):
        return None


def _alive(tags, year):
    """Was this way present in `year` per its OHM start/end dates?"""
    s, e = _yr(tags.get("start_date")), _yr(tags.get("end_date"))
    if s is None:               # undated -> cannot place it in time; drop
        return False
    return s <= year and (e is None or e >= year)


def ohm_query_lines(bbox_ll, year, timeout=240):
    """GeoDataFrame(EPSG:4326) of vehicular OHM streets ALIVE in `year` inside
    the lon/lat bbox (minx, miny, maxx, maxy).  Raw historical geometry, unclipped."""
    import json
    minx, miny, maxx, maxy = bbox_ll
    bb = f"{miny},{minx},{maxy},{maxx}"          # Overpass wants S,W,N,E
    q = f'[out:json][timeout:{timeout}];way["highway"]({bb});out geom;'
    els = None
    for attempt in range(4):
        res = subprocess.run(
            ["curl", "-s", "--max-time", str(timeout), "--retry", "3",
             "--retry-delay", "8", "--retry-all-errors", "-G", OHM_OVERPASS,
             "--data-urlencode", f"data={q}"],
            capture_output=True, text=True)
        try:
            els = json.loads(res.stdout).get("elements", [])
            break
        except json.JSONDecodeError:
            if attempt == 3:
                raise RuntimeError(
                    f"OHM Overpass returned no JSON after retries "
                    f"(bbox={bbox_ll}, year={year}); body[:200]={res.stdout[:200]!r}")
            import time
            time.sleep(30 * (attempt + 1))     # server throttles big back-to-back queries
    rows = []
    for e in els:
        t = e.get("tags", {})
        if t.get("highway") in _DROP_HWY:
            continue
        if not _alive(t, year):
            continue
        geom = e.get("geometry")
        if not geom or len(geom) < 2:
            continue
        line = LineString([(p["lon"], p["lat"]) for p in geom])
        rows.append(dict(name=t.get("name", ""), hwy=t.get("highway", ""),
                         start=t.get("start_date", ""), end=t.get("end_date", ""),
                         geometry=line))
    return gpd.GeoDataFrame(rows, crs=4326)


def fetch_ohm_year(slug, year, cache=True, tag=None, clip_geom=None, timeout=240):
    """Streets alive in `year` inside Ω (city UTM), cached to prehist/{slug}_{tag}.gpkg.
    tag defaults to the year; clip_geom (UTM) overrides Ω for a custom study area."""
    cfg = CITIES[slug]
    tag = tag or str(year)
    out = PREHIST_DIR / f"{slug}_{tag}.gpkg"
    if cache and out.exists():
        return gpd.read_file(out, layer="streets")
    omega = gpd.read_file(BND_DIR / f"{slug}.gpkg", layer="omega")
    clip_utm = clip_geom if clip_geom is not None else omega.to_crs(cfg["utm"]).geometry.iloc[0]
    minx, miny, maxx, maxy = gpd.GeoSeries([clip_utm], crs=cfg["utm"]).to_crs(4326).total_bounds
    g = ohm_query_lines((minx, miny, maxx, maxy), year, timeout).to_crs(cfg["utm"])
    g = gpd.clip(g, clip_utm.buffer(150.0))
    g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
    if cache:
        g.to_file(out, layer="streets")
    return g


def fetch_ohm_prehighway(slug, cache=True, timeout=240):
    """GeoDataFrame (city UTM) of streets alive at build_start-1 inside Ω.
    Cached to prehist/{slug}_pre.gpkg; pass cache=False to force a re-fetch."""
    cfg = CITIES[slug]
    year = cfg["build_start"] - 1
    out = PREHIST_DIR / f"{slug}_pre.gpkg"
    if cache and out.exists():
        return gpd.read_file(out, layer="pre_streets")

    omega = gpd.read_file(BND_DIR / f"{slug}.gpkg", layer="omega")
    omega_ll = omega.to_crs(4326)
    g = ohm_query_lines(tuple(omega_ll.total_bounds), year, timeout).to_crs(cfg["utm"])

    # clip to Ω (buffer a touch so boundary streets survive noding)
    om = omega.to_crs(cfg["utm"]).geometry.iloc[0]
    g = gpd.clip(g, om.buffer(150.0))
    g = g[~g.geometry.is_empty & g.geometry.notna()].reset_index(drop=True)
    if cache:
        g.to_file(out, layer="pre_streets")
    return g


def build_pre_graph(slug):
    """(G, pos, layers): largest connected component of the pre-highway street
    grid inside Ω, noded at intersections.  layers reuses the modern Ω / barrier
    / highway footprint so Regime-1 shares the same study geometry."""
    cfg = CITIES[slug]
    g = fetch_ohm_prehighway(slug)
    gpkg = BND_DIR / f"{slug}.gpkg"
    omega = gpd.read_file(gpkg, layer="omega").geometry.iloc[0]
    barrier = gpd.read_file(gpkg, layer="barrier").geometry.iloc[0]
    hwy = unary_union(gpd.read_file(gpkg, layer="highway").geometry.values)
    holc_d = unary_union(gpd.read_file(gpkg, layer="holc_d").geometry.values)

    G = _graph_from_lines(_noded_lines(g.geometry.values))
    if G.number_of_nodes() == 0:
        return G, {}, dict(omega=omega, barrier=barrier, highway=hwy, holc_d=holc_d)
    comp = max(nx.connected_components(G), key=len)
    G = G.subgraph(comp).copy()
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}
    return G, pos, dict(omega=omega, barrier=barrier, highway=hwy, holc_d=holc_d)
