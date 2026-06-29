#!/usr/bin/env python3
"""Stage 8b: build the tract-level analysis panel.

Joins the UOI measures (composite score + 6 spec metrics) to external
socioeconomic outcomes, all keyed on the 11-digit census-tract GEOID:

  Opportunity Atlas  economic mobility   kfr_pooled_pooled_p25, jail_*
  Eviction Lab       eviction pressure   filing_rate, judgement_rate (2014-18 mean)
  FARS               pedestrian safety   pooled ped fatalities 2017-21 (sjoin) + per-area
  LODES8             jobs                C000 (count), CE03/C000 (stable-job share), density
  ACS (optional)     race + education    added by 08b_acs.py if a Census key is supplied

Output: data/outputs/tract_panel.parquet
"""
import glob, zipfile, io
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = Path("/home/wnlab/CK_street")
EXT = ROOT / "data/external"
OUT = ROOT / "data/outputs"

def geoid(s):
    return s.astype("Int64").astype(str).str.zfill(11)

# ---------------- base: UOI score + 6 metrics ----------------
base = pd.read_csv(ROOT / "results/top1000/uoi_scores_all.csv")
base["GEOID"] = base["GEOID"].astype(str).str.zfill(11)
UOI_COLS = ["UOI_score", "link_node_ratio", "connected_node_ratio",
            "intersection_density", "median_block_length_ft",
            "walking_circuity", "pedshed_reach"]
panel = base[["GEOID"] + UOI_COLS].copy()
print(f"base UOI panel: {len(panel)} tracts")

# ---------------- tract geometry + ALAND (area, sqkm) ----------------
gdfs = []
for fp in sorted(glob.glob(str(ROOT / "data/tracts_*.gpkg"))):
    if "06075" in fp or "01ALL" in fp:      # skip pilot/dup layers
        continue
    g = gpd.read_file(fp)[["GEOID", "ALAND", "geometry"]]
    gdfs.append(g)
tracts = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
tracts = tracts.drop_duplicates("GEOID")
tracts["area_sqkm"] = tracts["ALAND"] / 1e6
print(f"tract geometries: {len(tracts)}")
area = tracts[["GEOID", "area_sqkm"]]

# ---------------- Opportunity Atlas ----------------
oa = pd.read_csv(EXT / "oa_tract_outcomes_simple.csv",
                 usecols=["state", "county", "tract",
                          "kfr_pooled_pooled_p25", "jail_pooled_pooled_p25"])
oa["GEOID"] = (oa["state"].astype(int).astype(str).str.zfill(2)
               + oa["county"].astype(int).astype(str).str.zfill(3)
               + oa["tract"].astype(int).astype(str).str.zfill(6))
oa = oa.rename(columns={"kfr_pooled_pooled_p25": "mobility_kfr_p25",
                        "jail_pooled_pooled_p25": "incarceration_p25"})
panel = panel.merge(oa[["GEOID", "mobility_kfr_p25", "incarceration_p25"]],
                    on="GEOID", how="left")
print(f"  + Opportunity Atlas: {panel['mobility_kfr_p25'].notna().sum()} matched")

# ---------------- Eviction Lab (2014-2018 mean of observed) ----------------
ev = pd.read_csv(EXT / "evictionlab_tract_2000_2018.csv",
                 usecols=["fips", "year", "type", "filing_rate", "judgement_rate"])
ev = ev[(ev["type"] == "observed") & (ev["year"].between(2014, 2018))]
ev["GEOID"] = geoid(ev["fips"])
evg = ev.groupby("GEOID")[["filing_rate", "judgement_rate"]].mean().reset_index()
evg = evg.rename(columns={"filing_rate": "eviction_filing_rate",
                          "judgement_rate": "eviction_judgement_rate"})
panel = panel.merge(evg, on="GEOID", how="left")
print(f"  + Eviction Lab: {panel['eviction_filing_rate'].notna().sum()} matched")

# ---------------- FARS pedestrian fatalities (pooled 2017-2021) ----------------
def read_member(zf, name):
    with zf.open(name) as fh:
        return pd.read_csv(io.BytesIO(fh.read()), encoding="latin-1", low_memory=False)

pts = []
for zp in sorted(glob.glob(str(EXT / "fars/FARS*.zip"))):
    with zipfile.ZipFile(zp) as zf:
        names = {n.lower().split("/")[-1]: n for n in zf.namelist()}
        acc = read_member(zf, names["accident.csv"])
        per = read_member(zf, names["person.csv"])
    acc.columns = [c.upper() for c in acc.columns]
    per.columns = [c.upper() for c in per.columns]
    ped = per[per["PER_TYP"] == 5].groupby("ST_CASE").size().rename("ped").reset_index()
    a = acc[["ST_CASE", "LATITUDE", "LONGITUD"]].merge(ped, on="ST_CASE")
    # drop FARS missing-coordinate sentinels
    a = a[(a["LATITUDE"] < 77) & (a["LONGITUD"].abs() < 777) & (a["LONGITUD"] != 0)]
    pts.append(a)
    print(f"  FARS {Path(zp).stem}: {int(a['ped'].sum())} ped fatalities w/ coords")
fars = pd.concat(pts, ignore_index=True)
fars_gdf = gpd.GeoDataFrame(fars,
            geometry=gpd.points_from_xy(fars["LONGITUD"], fars["LATITUDE"]),
            crs="EPSG:4326")
joined = gpd.sjoin(fars_gdf, tracts[["GEOID", "geometry"]], how="inner", predicate="within")
ped_by_tract = joined.groupby("GEOID")["ped"].sum().rename("ped_fatalities_5yr").reset_index()
panel = panel.merge(ped_by_tract, on="GEOID", how="left")
panel = panel.merge(area, on="GEOID", how="left")
panel["ped_fatalities_5yr"] = panel["ped_fatalities_5yr"].fillna(0.0)
panel["ped_fatal_per_100km2_yr"] = (panel["ped_fatalities_5yr"] / 5.0
                                    / panel["area_sqkm"] * 100)
print(f"  + FARS: {int(panel['ped_fatalities_5yr'].sum())} ped fatalities joined to tracts")

# ---------------- LODES8 WAC (jobs, 2021) ----------------
job = []
for gz in sorted(glob.glob(str(EXT / "lodes/*_wac_2021.csv.gz"))):
    d = pd.read_csv(gz, usecols=["w_geocode", "C000", "CE03"])
    d["GEOID"] = d["w_geocode"].astype("Int64").astype(str).str.zfill(15).str[:11]
    job.append(d.groupby("GEOID")[["C000", "CE03"]].sum())
jobs = pd.concat(job).groupby("GEOID").sum().reset_index()
jobs = jobs.rename(columns={"C000": "jobs_total", "CE03": "jobs_stable"})
jobs["stable_job_share"] = jobs["jobs_stable"] / jobs["jobs_total"].replace(0, np.nan)
panel = panel.merge(jobs[["GEOID", "jobs_total", "stable_job_share"]],
                    on="GEOID", how="left")
panel["job_density_per_sqkm"] = panel["jobs_total"] / panel["area_sqkm"]
print(f"  + LODES: {panel['jobs_total'].notna().sum()} matched")

# ---------------- save ----------------
panel.to_parquet(OUT / "tract_panel.parquet", index=False)
panel.to_csv(OUT / "tract_panel.csv", index=False)
print(f"\nsaved {len(panel)} tracts x {len(panel.columns)} cols -> tract_panel.parquet")
print(panel.drop(columns=["GEOID"]).describe().T[["count", "mean", "50%"]].round(3).to_string())
