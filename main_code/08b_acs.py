#!/usr/bin/env python3
"""Stage 8b (ACS add-on): fetch race + educational attainment per tract from the
Census ACS5 API and merge into data/outputs/tract_panel.parquet. Also adds a
population-normalized pedestrian-fatality rate (per 100k residents per yr).

Usage:
    python 08b_acs.py --key YOUR_CENSUS_API_KEY [--year 2022]
"""
import argparse, time, io
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path("/home/wnlab/CK_street")
OUT = ROOT / "data/outputs"

# 50 states + DC(11) + PR(72)
STATES = [f"{i:02d}" for i in range(1, 57) if i not in (3, 7, 14, 43, 52)] + ["72"]

VARS = ["B01001_001E",                     # total population
        "B02001_002E", "B02001_003E",      # white alone, black alone
        "B03003_001E", "B03003_003E",      # hisp denom, hispanic/latino
        "B15003_001E",                     # pop 25+ (education denom)
        "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]  # bachelor's+

def fetch(year, key):
    rows = []
    for st in STATES:
        url = (f"https://api.census.gov/data/{year}/acs/acs5"
               f"?get={','.join(VARS)}&for=tract:*&in=state:{st}&key={key}")
        for attempt in range(4):
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.text.lstrip().startswith("["):
                break
            time.sleep(2)
        else:
            print(f"  state {st}: FAILED ({r.status_code})"); continue
        j = r.json()
        df = pd.DataFrame(j[1:], columns=j[0])
        rows.append(df)
        print(f"  state {st}: {len(df)} tracts", flush=True)
    return pd.concat(rows, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--year", default="2022")
    a = ap.parse_args()

    raw = fetch(a.year, a.key)
    for v in VARS:
        raw[v] = pd.to_numeric(raw[v], errors="coerce")
    raw["GEOID"] = raw["state"] + raw["county"] + raw["tract"]

    pop = raw["B01001_001E"]
    acs = pd.DataFrame({"GEOID": raw["GEOID"], "population": pop})
    acs["pct_white"] = 100 * raw["B02001_002E"] / pop.replace(0, np.nan)
    acs["pct_black"] = 100 * raw["B02001_003E"] / pop.replace(0, np.nan)
    acs["pct_hispanic"] = 100 * raw["B03003_003E"] / raw["B03003_001E"].replace(0, np.nan)
    bach = raw[["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]].sum(axis=1)
    acs["pct_bachelor_plus"] = 100 * bach / raw["B15003_001E"].replace(0, np.nan)

    panel = pd.read_parquet(OUT / "tract_panel.parquet")
    panel = panel.drop(columns=[c for c in acs.columns if c != "GEOID" and c in panel], errors="ignore")
    panel = panel.merge(acs, on="GEOID", how="left")

    # population-normalized pedestrian fatality rate (per 100k residents per yr)
    if "ped_fatalities_5yr" in panel:
        panel["ped_fatal_per_100k_pop_yr"] = (panel["ped_fatalities_5yr"] / 5.0
                                              / panel["population"].replace(0, np.nan) * 1e5)

    panel.to_parquet(OUT / "tract_panel.parquet", index=False)
    panel.to_csv(OUT / "tract_panel.csv", index=False)
    print(f"\nmerged ACS into panel: {len(panel)} tracts x {len(panel.columns)} cols")
    print(f"  population matched: {panel['population'].notna().sum()}")
    print(panel[["pct_white", "pct_black", "pct_hispanic", "pct_bachelor_plus",
                 "ped_fatal_per_100k_pop_yr"]].describe().round(2).to_string())

if __name__ == "__main__":
    main()
