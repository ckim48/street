"""Shared configuration for the PathFinder highway-severance RJ-MCMC project.

Five deep-dive neighborhoods that a mid-century interstate cut through a
redlined (HOLC "D") Black community, and that today have active
"Reconnecting Communities" removal / cap / boulevard plans:

    Detroit      I-375   Black Bottom / Paradise Valley
    Syracuse     I-81    15th Ward
    New Orleans  I-10    Tremé / Claiborne
    St. Paul     I-94    Rondo
    Miami        I-95/395 Overtown

The project runs three RJ-MCMC "regimes" beside the national Tier-0 UOI work:
  R1  pre-highway (construction-year - 1) network -> plain optimal-network search
  R2  post-highway network with severance constraints -> PRE->POST hindcast
  R3  today's TIGER network -> budget/cost-constrained restoration (add-only)

Data layout (all under data/pathfinder/):
  tiger_roads/  per-county TIGER 2025 ROADS zips (modern network + highway lines)
  boundaries/   per-city Omega (study boundary) + barrier B + HOLC-D + highway
  graphs/       per-city decade / pre / post / modern noded graphs
Figures land in results/pathfinder/.

CHRONEX (dated roads for the pre/post networks) and the national HOLC polygons
are reused from the Tier-1 data tree (data/tier1/).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # repo root (CK_street/)
PF = ROOT / "data" / "pathfinder"
TIGER_ROADS = PF / "tiger_roads"
BND_DIR = PF / "boundaries"
GRAPH_DIR = PF / "graphs"
LOG_DIR = PF / "logs"
CHRONEX_DIR = ROOT / "data" / "tier1" / "chronex"       # reuse Tier-1 CHRONEX tree
HOLC_GPKG = ROOT / "data" / "tier1" / "holc" / "mappinginequality.gpkg"
RES = ROOT / "results" / "pathfinder"
for _d in (PF, TIGER_ROADS, BND_DIR, GRAPH_DIR, LOG_DIR, RES):
    _d.mkdir(parents=True, exist_ok=True)

# ---- the five severed neighborhoods --------------------------------------
# anchor = (lon, lat) approximate historic neighborhood centre, used to localize
#   the HOLC-D selection to THIS neighborhood (the interstate itself runs the
#   whole county).  Refine against the verification figure if a city grabs the
#   wrong redlined polygons.
# fullnames = TIGER ROADS FULLNAME values of the target interstate (S1100).
# row_width = full highway right-of-way width in metres (barrier B = buffer w/2);
#   from the severance spec: I-375 trench ~90, I-81 viaduct ~40, I-10 elevated
#   ~40, I-94 trench ~75, I-95/395 interchange ~100.
# build_start = construction start year; the pre-highway snapshot uses build_start-1.
# cbsa = CHRONEX-US per-metro GeoPackage id (chronex_us_<cbsa>.gpkg).
# utm = projected CRS (metre geometry) for this city.
# project_cost = headline Reconnecting-Communities program cost (USD), the R3
#   budget anchor; None where no firm public figure was found.
CITIES = {
    "detroit": dict(
        city="Detroit", state="MI", holc_match="Detroit", county="26163",
        cbsa="19820", utm=32617, highway="I-375", fullnames=["I- 375"],
        neighborhood="Black Bottom / Paradise Valley",
        anchor=(-83.0385, 42.3455), build_start=1959, row_width=90.0,
        project_cost=270_000_000,
        cost_note="I-375 trench-to-boulevard conversion (MDOT); $104.6M federal RCP grant"),
    "syracuse": dict(
        city="Syracuse", state="NY", holc_match="Syracuse", county="36067",
        cbsa="45060", utm=32618, highway="I-81", fullnames=["I- 81"],
        neighborhood="15th Ward",
        anchor=(-76.1430, 43.0480), build_start=1964, row_width=40.0,
        project_cost=2_250_000_000,
        cost_note="I-81 viaduct removal + community grid; NYSDOT, $180M federal RCP"),
    "new_orleans": dict(
        city="New Orleans", state="LA", holc_match="New Orleans", county="22071",
        cbsa="35380", utm=32615, highway="I-10 (Claiborne)", fullnames=["I- 10"],
        neighborhood="Treme / Claiborne",
        anchor=(-90.0700, 29.9720), build_start=1966, row_width=40.0,
        project_cost=95_000_000,
        cost_note="Claiborne ramp removal / at-grade plan ($95M); full removal $4B+"),
    "st_paul": dict(
        city="St. Paul", state="MN", holc_match="Paul", county="27123",
        cbsa="33460", utm=32615, highway="I-94", fullnames=["I- 94"],
        neighborhood="Rondo",
        anchor=(-93.1280, 44.9510), build_start=1956, row_width=75.0,
        project_cost=450_000_000,
        cost_note="ReConnect Rondo land bridge over I-94 (~$450M program)"),
    "miami": dict(
        city="Miami", state="FL", holc_match="Miami", county="12086",
        cbsa="33100", utm=32617, highway="I-95 / I-395", fullnames=["I- 95", "I- 395"],
        neighborhood="Overtown",
        anchor=(-80.1990, 25.7870), build_start=1959, row_width=100.0,
        project_cost=None,
        cost_note="I-395 signature bridge / Connecting Miami; no firm removal figure"),
    "kansas_city": dict(
        city="Kansas City", state="MO", holc_match="Kansas City", county="29095",
        cbsa="28140", utm=32615, highway="I-70", fullnames=["I- 70"],
        neighborhood="18th & Vine",
        anchor=(-94.5580, 39.0920), build_start=1956, row_width=70.0,
        project_cost=None,
        cost_note="I-70 severed the redlined 18th & Vine jazz district; no firm removal figure"),
}

CITY_SLUGS = list(CITIES.keys())

# HOLC grade that marks the redlined (treated) neighborhood fabric.
HOLC_D_GRADE = "D"
HOLC_D_CATEGORY = "Hazardous"

# localization / selection buffers (metres), used by 00_boundaries.py
LOCALIZE_R = 2500.0      # keep highway + HOLC-D within this radius of the anchor
SELECT_BUF = 450.0       # a HOLC-D polygon counts as "cut" if within this of the highway
MODERN_MARGIN = 150.0    # clip modern TIGER roads to Omega buffered by this


def holc_d_for(g, cfg):
    """Filter the national HOLC GeoDataFrame to this city's grade-D polygons."""
    m = (g["city"].astype(str).str.contains(cfg["holc_match"], case=False, na=False)
         & g["state"].astype(str).str.upper().eq(cfg["state"]))
    sub = g[m]
    d = sub[sub["grade"].astype(str).str.upper().str.strip().eq(HOLC_D_GRADE)]
    if len(d) == 0 and "category" in sub.columns:
        d = sub[sub["category"].astype(str).str.strip().eq(HOLC_D_CATEGORY)]
    return d
