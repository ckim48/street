#!/usr/bin/env python3
"""Stage 8d: adjusted UOI-outcome regressions.

Per outcome Y, standardized OLS:
    Y ~ UOI_score + log10(pop_density) + log10(median_income) + pct_white
        + state fixed effects
Reports the standardized UOI coefficient (t, p) beside the raw bivariate
Spearman. OLS with analytic SEs in numpy.

Outputs (results/external_correlates/):
  regression_uoi_adjusted.csv   raw r vs adjusted beta_UOI (+ t, p, n) per outcome
  fig_raw_vs_adjusted.png       raw vs adjusted UOI effect
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/wnlab/CK_street")
RES = ROOT / "results/external_correlates"

p = pd.read_parquet(ROOT / "data/outputs/tract_panel.parquet")
p = p.replace([np.inf, -np.inf], np.nan)
p["state"] = p["GEOID"].str[:2]
p["pop_density"] = p["population"] / p["area_sqkm"]

OUT = [
    ("mobility_kfr_p25",          "Econ. mobility (p25)"),
    ("incarceration_p25",         "Incarceration (p25)"),
    ("eviction_filing_rate",      "Eviction filing rate"),
    ("ped_fatal_per_100k_pop_yr", "Pedestrian fatalities /100k pop"),
    ("stable_job_share",          "Stable-job share"),
    ("job_density_per_sqkm",      "Job density"),
    ("pct_bachelor_plus",         "Bachelor's+ share"),
]
CONTROLS = ["log_density", "log_income", "pct_white"]

def z(s):
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

def ols(y, X):
    """return beta, se, t, p, n for design X (incl intercept). y,X aligned, finite."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = n - k
    sigma2 = (resid @ resid) / dof
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    t = beta / se
    pval = 2 * stats.t.sf(np.abs(t), dof)
    return beta, se, t, pval, n

rows = []
for col, lab in OUT:
    d = p[[col, "UOI_score", "pop_density", "median_income", "pct_white", "state"]].copy()
    # winsorize heavy-tailed outcome for stability
    lo, hi = d[col].quantile([0.005, 0.995])
    d[col] = d[col].clip(lo, hi)
    # per-capita ped rate: drop tiny-population tracts (rate is noise there)
    if col == "ped_fatal_per_100k_pop_yr":
        d = d[p["population"] >= 200]
    d["log_density"] = np.log10(d["pop_density"].clip(lower=1e-3))
    d["log_income"] = np.log10(d["median_income"])
    d = d.dropna(subset=[col, "UOI_score"] + ["pop_density", "median_income", "pct_white"])
    if len(d) < 200:
        continue
    # raw bivariate (Spearman)
    raw = stats.spearmanr(d["UOI_score"], d[col]).statistic
    # standardized design: intercept + UOI + controls + state dummies
    y = z(d[col]).values
    feats = pd.DataFrame({"UOI_score": z(d["UOI_score"]),
                          "log_density": z(d["log_density"]),
                          "log_income": z(d["log_income"]),
                          "pct_white": z(d["pct_white"])})
    st = pd.get_dummies(d["state"], drop_first=True).astype(float).reset_index(drop=True)
    X = np.column_stack([np.ones(len(d)), feats.reset_index(drop=True).values, st.values])
    beta, se, t, pval, n = ols(y, X)
    # index 1 = UOI_score (0 is intercept)
    rows.append({"outcome": lab, "raw_spearman": round(raw, 3),
                 "adj_beta_UOI": round(float(beta[1]), 3),
                 "t": round(float(t[1]), 1), "p": float(pval[1]), "n": n})
    print(f"{lab:32s} raw rho={raw:+.3f}  adj betaUOI={beta[1]:+.3f}  "
          f"t={t[1]:+.1f}  p={pval[1]:.1e}  n={n}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(RES / "regression_uoi_adjusted.csv", index=False)

# ---------------- figure: raw vs adjusted UOI effect ----------------
fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
yy = np.arange(len(res))
ax.barh(yy - 0.2, res["raw_spearman"], 0.4, label="raw bivariate (Spearman rho)", color="#999999")
ax.barh(yy + 0.2, res["adj_beta_UOI"], 0.4,
        label="adjusted (density+income+race+state FE)", color="#C44E52")
ax.set_yticks(yy); ax.set_yticklabels(res["outcome"]); ax.invert_yaxis()
ax.axvline(0, color="k", lw=0.8)
for i, r in res.iterrows():
    sig = "***" if r["p"] < 1e-3 else "**" if r["p"] < 1e-2 else "*" if r["p"] < 0.05 else "ns"
    ax.text(r["adj_beta_UOI"] + (0.005 if r["adj_beta_UOI"] >= 0 else -0.005), i + 0.2,
            sig, va="center", ha="left" if r["adj_beta_UOI"] >= 0 else "right", fontsize=8)
ax.set_xlabel("UOI effect size (standardized)")
ax.set_title("UOI vs outcomes: raw correlation vs effect adjusted for\n"
             "population density + income + %white + state fixed effects")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(RES / "fig_raw_vs_adjusted.png", dpi=140)
print(f"\nsaved {RES/'fig_raw_vs_adjusted.png'}")
print(res.to_string(index=False))
