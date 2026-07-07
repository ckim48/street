"""Tier 1 - Step 1+2: per-city HOLC boundary inventory.

For each of the six Tier 1 cities this script:
  1. subsets the national Mapping Inequality gpkg to the city, cleans grades;
  2. dissolves polygons to one region per grade;
  3. extracts the shared C-D and B-C boundary lines (the RD frontier) with a
     small buffer so hand-drawn HOLC polygons that only *nearly* touch still
     register as adjacent;
  4. cuts each frontier into fixed-length segments -> the boundary-segment
     fixed-effect units the RD uses;
  5. classifies each segment as street / rail / water / freeway / harbor /
     ambiguous by overlaying OSM barrier features (Guide Step 2.3), so
     lake/river/rail/freeway segments can be dropped from the clean RD sample.

Outputs (per city, data/tier1/boundaries/):
  {slug}_grades.gpkg        dissolved grade regions (layer 'grades')
  {slug}_boundaries.gpkg    segmented boundary lines (layer 'segments')
  {slug}_inventory.csv      per-(pair,barrier) km + segment counts (city log)
  ../boundaries/inventory_all.csv   all cities stacked

Usage:
  python tier1/01_holc_boundaries.py                 # all cities
  python tier1/01_holc_boundaries.py --cities chicago detroit
  python tier1/01_holc_boundaries.py --seg-len 150 --no-osm   # skip OSM barriers
"""
from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, unary_union

from tier1_common import (BND_DIR, BOUNDARY_PAIRS, CITIES, HOLC_GPKG, METRIC_CRS,
                          clean_grade, city_slugs)

warnings.filterwarnings("ignore", category=UserWarning)

ADJ_BUFFER_M = 30.0        # two grades count as adjacent within this gap
SEG_LEN_M = 150.0          # default boundary-segment length (RD FE unit)
OSM_NEAR_M = 25.0          # a segment is "on" a barrier within this distance


# ------------------------------------------------------------------ geometry
def load_city(slug: str) -> gpd.GeoDataFrame:
    cfg = CITIES[slug]
    g = gpd.read_file(HOLC_GPKG)
    g = g[(g.city == cfg["city"]) & (g.state == cfg["state"])].copy()
    g["grade"] = g["grade"].map(clean_grade)
    g = g[g.grade.isin(list("ABCD"))]
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    return g.to_crs(METRIC_CRS)


def dissolve_grades(city: gpd.GeoDataFrame) -> dict[str, "gpd.GeoSeries"]:
    out = {}
    for grade, sub in city.groupby("grade"):
        out[grade] = unary_union(sub.geometry.buffer(0).values)   # buffer(0) fixes self-int
    return out


def shared_frontier(hi_geom, lo_geom):
    """Line where the higher-grade region abuts the lower-grade region."""
    if hi_geom is None or lo_geom is None:
        return None
    near = hi_geom.boundary.intersection(lo_geom.buffer(ADJ_BUFFER_M))
    if near.is_empty:
        return None
    # keep only 1-D pieces
    lines = [g for g in getattr(near, "geoms", [near])
             if g.geom_type in ("LineString", "MultiLineString") and not g.is_empty]
    if not lines:
        return None
    merged = linemerge(unary_union(lines))
    return merged if not merged.is_empty else None


def cut_segments(line, seg_len: float) -> list[LineString]:
    """Split a (Multi)LineString into ~seg_len pieces by arc length."""
    parts = list(getattr(line, "geoms", [line]))
    segs = []
    for part in parts:
        L = part.length
        if L < seg_len * 0.5:
            segs.append(part)
            continue
        n = max(1, int(round(L / seg_len)))
        cuts = np.linspace(0, L, n + 1)
        for a, b in zip(cuts[:-1], cuts[1:]):
            pa, pb = part.interpolate(a), part.interpolate(b)
            # densify the sub-segment so it follows the frontier's curvature
            mids = [part.interpolate(t) for t in np.linspace(a, b, 8)]
            segs.append(LineString([pa, *mids, pb]))
    return [s for s in segs if s.length > 1.0]


# ------------------------------------------------------------- OSM barriers
def osm_barriers(city: gpd.GeoDataFrame):
    """Fetch OSM water / rail / freeway geometries around the city (metric CRS).
    Returns dict class -> unary_union geometry (or None on failure)."""
    import osmnx as ox
    hull = city.to_crs(4326).union_all().convex_hull
    tagsets = {
        "water":    {"natural": "water", "waterway": True, "water": True},
        "harbor":   {"natural": "coastline", "harbour": True, "landuse": "harbour"},
        "rail":     {"railway": ["rail", "light_rail", "subway", "tram"]},
        "freeway":  {"highway": ["motorway", "motorway_link", "trunk", "trunk_link"]},
    }
    out = {}
    for cls, tags in tagsets.items():
        try:
            f = ox.features_from_polygon(hull, tags)
            if len(f):
                out[cls] = unary_union(f.to_crs(METRIC_CRS).geometry.values)
            else:
                out[cls] = None
        except Exception as e:                                    # noqa: BLE001
            print(f"    [osm] {cls}: {type(e).__name__} {e}")
            out[cls] = None
    return out


def classify(seg: LineString, barriers: dict | None) -> str:
    if barriers is None:
        return "ambiguous"
    for cls in ("water", "harbor", "rail", "freeway"):   # water/harbor win ties
        geom = barriers.get(cls)
        if geom is not None and seg.distance(geom) < OSM_NEAR_M:
            return cls
    return "street"


# ------------------------------------------------------------------- driver
def process_city(slug: str, seg_len: float, use_osm: bool) -> pd.DataFrame:
    cfg = CITIES[slug]
    print(f"[{slug}] {cfg['city']}, {cfg['state']}")
    city = load_city(slug)
    print(f"  {len(city)} graded polygons")
    regions = dissolve_grades(city)

    # save dissolved grade regions for maps
    greg = gpd.GeoDataFrame(
        {"grade": list(regions)}, geometry=list(regions.values()), crs=METRIC_CRS)
    greg.to_file(BND_DIR / f"{slug}_grades.gpkg", layer="grades", driver="GPKG")

    barriers = osm_barriers(city) if use_osm else None
    if use_osm:
        have = [k for k, v in (barriers or {}).items() if v is not None]
        print(f"  OSM barriers: {', '.join(have) if have else 'none'}")

    rows, geoms = [], []
    for hi, lo in BOUNDARY_PAIRS:
        line = shared_frontier(regions.get(hi), regions.get(lo))
        if line is None:
            print(f"  {hi}-{lo}: no shared frontier")
            continue
        segs = cut_segments(line, seg_len)
        for i, s in enumerate(segs):
            bcls = classify(s, barriers) if use_osm else "unclassified"
            rows.append(dict(slug=slug, city=cfg["city"], pair=f"{hi}-{lo}",
                             hi_grade=hi, lo_grade=lo,
                             seg_id=f"{slug}_{hi}{lo}_{i:04d}",
                             length_m=round(s.length, 1), barrier=bcls))
            geoms.append(s)
        km = sum(r["length_m"] for r in rows if r["pair"] == f"{hi}-{lo}") / 1000
        print(f"  {hi}-{lo}: {len(segs)} segments, {km:.1f} km")

    if not rows:
        return pd.DataFrame()
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=METRIC_CRS)
    gdf.to_file(BND_DIR / f"{slug}_boundaries.gpkg", layer="segments", driver="GPKG")

    # per-(pair,barrier) inventory = the guide's city-log boundary-priority table
    inv = (gdf.groupby(["pair", "barrier"])
              .agg(n_seg=("seg_id", "size"), km=("length_m", lambda s: s.sum()/1000))
              .reset_index())
    inv.insert(0, "slug", slug)
    inv["km"] = inv["km"].round(2)
    inv.to_csv(BND_DIR / f"{slug}_inventory.csv", index=False)
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=city_slugs(),
                    help="subset of city slugs (default: all six)")
    ap.add_argument("--seg-len", type=float, default=SEG_LEN_M)
    ap.add_argument("--no-osm", action="store_true",
                    help="skip OSM barrier classification (offline / faster)")
    args = ap.parse_args()

    all_inv = []
    for slug in args.cities:
        try:
            inv = process_city(slug, args.seg_len, use_osm=not args.no_osm)
            if len(inv):
                all_inv.append(inv)
        except Exception as e:                                    # noqa: BLE001
            print(f"[{slug}] FAILED: {type(e).__name__}: {e}")
    if all_inv:
        allc = pd.concat(all_inv, ignore_index=True)
        allc.to_csv(BND_DIR / "inventory_all.csv", index=False)
        print(f"\nsaved -> {BND_DIR/'inventory_all.csv'}")
        print(allc.to_string(index=False))


if __name__ == "__main__":
    main()
