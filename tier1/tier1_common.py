"""Shared configuration and helpers for the Tier 1 HOLC boundary RD experiment.

Tier 1 is the city-level deep dive that sits *beside* the national Tier 0 UOI
result: for six HOLC-mapped cities it estimates a geographic regression
discontinuity (RD) in street-network organization (the six-metric UOI / "Oi"
index) across HOLC grade boundaries, plus the mechanism narratives and visual
validation panels described in the Tier 1 Experiment Guide.

Data layout (all under data/tier1/):
  holc/        national Mapping Inequality gpkg + per-city subsets
  boundaries/  per-city C-D and B-C boundary inventories (GeoPackage + CSV)
  graphs/      per-city decade road graphs (CHRONEX-US derived) + present-day
  oi/          per-boundary-point Oi (UOI) tables, per decade
  rd/          per-city RD estimation tables
Figures land in results/tier1/.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # repo root (CK_street/)
T1 = ROOT / "data" / "tier1"
HOLC_DIR = T1 / "holc"
BND_DIR = T1 / "boundaries"
GRAPH_DIR = T1 / "graphs"
OI_DIR = T1 / "oi"
RD_DIR = T1 / "rd"
RES = ROOT / "results" / "tier1"
for _d in (HOLC_DIR, BND_DIR, GRAPH_DIR, OI_DIR, RD_DIR, RES):
    _d.mkdir(parents=True, exist_ok=True)

HOLC_GPKG = HOLC_DIR / "mappinginequality.gpkg"

# ---- the six Tier 1 cities: (city, state) exactly as they appear in the ------
# Mapping Inequality `city`/`state` columns, plus HOLC map year, role, and the
# guide's watch-out.  `slug` is the short id used in filenames.
# `cbsa` = 5-digit Census 2019 CBSA GEOID; it is the CHRONEX-US per-metro
# GeoPackage suffix (chronex_us_<cbsa>.gpkg).  Detroit/Atlanta/LA codes are the
# standard 2019 definitions and are re-verified against the ZIP file list at
# fetch time (00_fetch_tier1.sh).
CITIES = {
    "chicago":      dict(city="Chicago",      state="IL", holc_year=1940, cbsa="16980",
                         role="Benchmark: large C-D stock on a uniform grid",
                         watch_out="Drop lake & river segments"),
    "philadelphia": dict(city="Philadelphia", state="PA", holc_year=1937, cbsa="37980",
                         role="Uniform rowhouse grid; strong continuity",
                         watch_out="Flag rail-line boundaries"),
    "baltimore":    dict(city="Baltimore",    state="MD", holc_year=1937, cbsa="12580",
                         role="Rowhouse fabric; redlining-literature overlap",
                         watch_out="Flag harbor-adjacent segments"),
    "detroit":      dict(city="Detroit",      state="MI", holc_year=1939, cbsa="19820",
                         role="Barrier-robustness showcase; post-1950 divergence",
                         watch_out="Use as robustness, not headline"),
    "atlanta":      dict(city="Atlanta",      state="GA", holc_year=1938, cbsa="12060",
                         role="Sunbelt growth; widening-gap trajectory",
                         watch_out="Topography & less regular fabric"),
    "los_angeles":  dict(city="Los Angeles",  state="CA", holc_year=1939, cbsa="31080",
                         role="Freeway-mechanism chapter; large western polygons",
                         watch_out="Treat freeways as post-treatment barriers"),
}

# The RD boundary pairs the guide asks for: (higher grade, lower=treated grade).
# Treatment is always the LOWER-grade (worse-rated) side of the boundary.
BOUNDARY_PAIRS = [("C", "D"), ("B", "C")]

# Decade snapshots for the Oi-by-decade trajectory (guide Step 3).
DECADES = [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

# Barrier classes (guide Step 2.3).  A boundary segment that runs along one of
# these is flagged; street/administrative segments are the clean RD sample.
BARRIER_CLASSES = ["street", "rail", "water", "freeway", "topography",
                   "harbor", "ambiguous"]

# Projected CRS per city for metric (meter) geometry work.  US National Atlas
# Equal Area (EPSG:5070) is fine for all six CONUS cities; kept explicit so a
# future non-CONUS city can override.
METRIC_CRS = 5070


def clean_grade(g) -> str:
    """Normalize the HOLC `grade` column ('A ', ' C', '') -> 'A'..'D' or ''."""
    return str(g).strip().upper() if g is not None else ""


def city_slugs() -> list[str]:
    return list(CITIES)
