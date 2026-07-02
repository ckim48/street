#!/usr/bin/env python3
"""Stage 8e: pedestrian-safety follow-up on the adjusted UOI effect from 08d.

  (1) exposure test: refit 08d ped model with ACS walk-to-work share added
      to the controls
  (2) component decomposition: adjusted effect of each of the 6 UOI metrics
      on per-capita ped fatality rate (same controls)
  (3) FARS crash profile by tract UOI quintile: road functional class,
      urban/rural, intersection vs mid-block

Outputs (results/external_correlates/):
  ped_exposure_test.csv, ped_component_effects.csv, ped_fars_profile.csv,
  fig_ped_safety.png
"""
import zipfile, io, glob, time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/wnlab/CK_street")
EXT = ROOT / "data/external"
RES = ROOT / "results/external_correlates"
KEY = "799d4507ebcc93a0f1199a84c3b85a555531fca9"
STATES = [f"{i:02d}" for i in range(1, 57) if i not in (3, 7, 14, 43, 52)] + ["72"]

p = pd.read_parquet(ROOT / "data/outputs/tract_panel.parquet")
p = p.replace([np.inf, -np.inf], np.nan)
p["state"] = p["GEOID"].str[:2]
p["pop_density"] = p["population"] / p["area_sqkm"]

# ---------- fetch ACS walk-to-work share (cache into panel) ----------
if "walk_share" not in p.columns:
    print("fetching ACS walk-to-work share ...", flush=True)
    rows = []
    for st in STATES:
        url = (f"https://api.census.gov/data/2022/acs/acs5"
               f"?get=B08301_001E,B08301_019E&for=tract:*&in=state:{st}&key={KEY}")
        for _ in range(4):
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and r.text.lstrip().startswith("["):
                break
            time.sleep(2)
        j = r.json(); d = pd.DataFrame(j[1:], columns=j[0])
        rows.append(d)
    w = pd.concat(rows, ignore_index=True)
    w["GEOID"] = w["state"] + w["county"] + w["tract"]
    w["tot"] = pd.to_numeric(w["B08301_001E"], errors="coerce")
    w["walk"] = pd.to_numeric(w["B08301_019E"], errors="coerce")
    w["walk_share"] = 100 * w["walk"] / w["tot"].replace(0, np.nan)
    p = p.merge(w[["GEOID", "walk_share"]], on="GEOID", how="left")
    p.to_parquet(ROOT / "data/outputs/tract_panel.parquet", index=False)
    print(f"  walk_share matched: {p['walk_share'].notna().sum()}", flush=True)

# ---------- OLS helper ----------
def z(s): return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

def ols_beta(y, X):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), dof)
    return beta, se, t, pv

def fit(df, target, predictor, controls):
    d = df.dropna(subset=[target, predictor] + controls).copy()
    y = z(d[target]).values
    cols = [predictor] + controls
    feats = np.column_stack([z(d[c]).values for c in cols])
    st = pd.get_dummies(d["state"], drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(d)), feats, st])
    beta, se, t, pv = ols_beta(y, X)
    return beta[1], t[1], pv[1], len(d)   # predictor is column index 1

# base frame for ped models: population floor + transforms
base = p[p["population"] >= 200].copy()
base["log_density"] = np.log10(base["pop_density"].clip(lower=1e-3))
base["log_income"] = np.log10(base["median_income"])
TARGET = "ped_fatal_per_100k_pop_yr"
lo, hi = base[TARGET].quantile([0.005, 0.995])
base[TARGET] = base[TARGET].clip(lo, hi)
CTRL = ["log_density", "log_income", "pct_white"]

# ---------- (1) exposure test ----------
b0, t0, p0, n0 = fit(base, TARGET, "UOI_score", CTRL)
b1, t1, p1, n1 = fit(base, TARGET, "UOI_score", CTRL + ["walk_share"])
bw, tw, pw, nw = fit(base, TARGET, "walk_share", CTRL + ["UOI_score"])
exp = pd.DataFrame([
    {"model": "UOI | density+income+race+stateFE",            "beta_UOI": round(b0, 3), "t": round(t0, 1), "n": n0},
    {"model": "UOI | + walk-to-work share (exposure)",        "beta_UOI": round(b1, 3), "t": round(t1, 1), "n": n1},
])
exp.to_csv(RES / "ped_exposure_test.csv", index=False)
print("\n(1) EXPOSURE TEST"); print(exp.to_string(index=False))
print(f"    walk_share's own adjusted beta = {bw:+.3f} (t={tw:+.1f})")
shrink = (1 - b1 / b0) * 100 if b0 else float("nan")
print(f"    -> adding walk share shrinks UOI effect by {shrink:.0f}%")

# ---------- (2) component decomposition ----------
comp = []
for m in ["UOI_score", "link_node_ratio", "connected_node_ratio",
          "intersection_density", "median_block_length_ft",
          "walking_circuity", "pedshed_reach"]:
    b, t, pv, n = fit(base, TARGET, m, CTRL)
    comp.append({"metric": m, "adj_beta": round(b, 3), "t": round(t, 1), "p": pv, "n": n})
comp = pd.DataFrame(comp)
comp.to_csv(RES / "ped_component_effects.csv", index=False)
print("\n(2) COMPONENT EFFECTS on per-capita ped fatality (adjusted)")
print(comp.to_string(index=False))

# ---------- (3) FARS mechanism profile by UOI quintile ----------
# load tract polygons
gdfs = []
for fp in sorted(glob.glob(str(ROOT / "data/tracts_*.gpkg"))):
    if "06075" in fp or "01ALL" in fp:
        continue
    gdfs.append(gpd.read_file(fp)[["GEOID", "geometry"]])
tracts = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
tracts = tracts.drop_duplicates("GEOID")

KEEP = ["ST_CASE", "LATITUDE", "LONGITUD", "RUR_URBNAME", "FUNC_SYSNAME", "RELJCT2NAME"]
pts = []
for zp in sorted(glob.glob(str(EXT / "fars/FARS*.zip"))):
    with zipfile.ZipFile(zp) as zf:
        nm = {n.lower().split("/")[-1]: n for n in zf.namelist()}
        acc = pd.read_csv(io.BytesIO(zf.open(nm["accident.csv"]).read()),
                          encoding="latin-1", low_memory=False)
        per = pd.read_csv(io.BytesIO(zf.open(nm["person.csv"]).read()),
                          encoding="latin-1", low_memory=False)
    acc.columns = [c.upper() for c in acc.columns]; per.columns = [c.upper() for c in per.columns]
    ped = per[per["PER_TYP"] == 5].groupby("ST_CASE").size().rename("ped").reset_index()
    a = acc[[c for c in KEEP if c in acc.columns]].merge(ped, on="ST_CASE")
    a = a[(a["LATITUDE"] < 77) & (a["LONGITUD"].abs() < 777) & (a["LONGITUD"] != 0)]
    pts.append(a)
fars = pd.concat(pts, ignore_index=True)
fg = gpd.GeoDataFrame(fars, geometry=gpd.points_from_xy(fars["LONGITUD"], fars["LATITUDE"]),
                      crs="EPSG:4326")
j = gpd.sjoin(fg, tracts, how="inner", predicate="within")
j = j.merge(p[["GEOID", "UOI_score"]], on="GEOID", how="left").dropna(subset=["UOI_score"])
j["uoi_q"] = pd.qcut(j["UOI_score"], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])

def share(colname, pos_substrings):
    s = j[colname].astype(str).str.lower()
    hit = s.apply(lambda v: any(k in v for k in pos_substrings))
    return j.assign(_h=hit).groupby("uoi_q", observed=True)["_h"].mean() * 100

prof = pd.DataFrame({
    "ped_deaths": j.groupby("uoi_q", observed=True)["ped"].sum(),
    "pct_urban": share("RUR_URBNAME", ["urban"]),
    "pct_at_intersection": share("RELJCT2NAME", ["intersection"]),
    "pct_arterial_or_higher": share("FUNC_SYSNAME", ["arterial", "interstate", "freeway", "expressway"]),
    "pct_local_road": share("FUNC_SYSNAME", ["local"]),
}).reset_index()
prof.to_csv(RES / "ped_fars_profile.csv", index=False)
print("\n(3) FARS ped-fatality profile by tract UOI quintile")
print(prof.round(1).to_string(index=False))

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")
ax[0].barh(comp["metric"], comp["adj_beta"],
           color=["#C44E52" if b > 0 else "#4C72B0" for b in comp["adj_beta"]])
ax[0].axvline(0, color="k", lw=0.8); ax[0].invert_yaxis()
ax[0].set_title("(2) Which UOI metric drives ped risk?\nadjusted beta on per-capita ped fatality")
ax[0].set_xlabel("standardized effect")

x = np.arange(len(prof))
ax[1].bar(x, prof["pct_at_intersection"], 0.4, label="at intersection", color="#DD8452")
ax[1].bar(x, prof["pct_arterial_or_higher"], 0.4, bottom=0, alpha=0,)  # spacer no-op
ax[1].plot(x, prof["pct_arterial_or_higher"], "o-", color="#C44E52", label="on arterial+")
ax[1].plot(x, prof["pct_urban"], "s-", color="#55A868", label="urban")
ax[1].set_xticks(x); ax[1].set_xticklabels(prof["uoi_q"], fontsize=8)
ax[1].set_ylabel("% of ped deaths"); ax[1].set_ylim(0, 100)
ax[1].set_title("(3) Where ped deaths happen, by UOI quintile"); ax[1].legend(fontsize=8)

ax[2].axis("off")
txt = ("(1) EXPOSURE TEST\n\n"
       f"UOI effect (adj.):      {b0:+.3f}  (t={t0:+.0f})\n"
       f"+ walk-to-work share:   {b1:+.3f}  (t={t1:+.0f})\n"
       f"walk-share own effect:  {bw:+.3f}  (t={tw:+.0f})\n\n"
       f"Adding walking exposure shrinks\nthe UOI effect by {shrink:.0f}%.\n\n"
       "If most of the UOI->fatality link is\nexposure (more walkers), the streets\n"
       "are busy, not intrinsically deadlier.")
ax[2].text(0.02, 0.95, txt, va="top", ha="left", fontsize=11, family="monospace")
fig.suptitle("Pedestrian-safety deep dive: is high-UOI deadly, or just busy?", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(RES / "fig_ped_safety.png", dpi=140)
print(f"\nsaved {RES/'fig_ped_safety.png'}")
