"""Shared paths and helpers for the UOI pipeline."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TIGER_DIR = DATA / "tiger"
GRAPH_DIR = DATA / "graphs"
CACHE_DIR = DATA / "osmnx_cache"
OUT_DIR = DATA / "outputs"

for d in (TIGER_DIR, GRAPH_DIR, CACHE_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# TIGER/Line census tract boundaries, one file per state
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state}_tract.zip"


def tiger_tracts(state: str, year: int = 2024):
    """Download (if needed) and load TIGER tract boundaries for a state FIPS code."""
    import geopandas as gpd
    import requests

    zip_path = TIGER_DIR / f"tl_{year}_{state}_tract.zip"
    if not zip_path.exists():
        url = TIGER_URL.format(year=year, state=state)
        print(f"Downloading {url}")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
    gdf = gpd.read_file(f"zip://{zip_path}")
    return gdf.to_crs(epsg=4326)


def graph_path(geoid: str) -> Path:
    return GRAPH_DIR / f"{geoid}.graphml"


def gini(x) -> float:
    """Gini coefficient of a 1-D array of non-negative values."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)
