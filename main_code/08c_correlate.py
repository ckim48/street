#!/usr/bin/env python3
"""Stage 8c: correlation / relationship charts between UOI and external outcomes.

Reads data/outputs/tract_panel.parquet and writes to results/external_correlates/:
  corr_matrix_spearman.csv      Spearman rho: UOI measures x outcomes (+ pairwise n)
  fig_corr_heatmap.png          the same matrix as an annotated heatmap
  fig_scatter_uoi_vs_outcomes.png   UOI_score vs each outcome (hexbin + trend + rho)
Spearman (rank) is used throughout: the variables are heavily skewed and the
relationships are monotonic-but-nonlinear, so Pearson would be misleading.
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
RES.mkdir(parents=True, exist_ok=True)

p = pd.read_parquet(ROOT / "data/outputs/tract_panel.parquet")
p = p.replace([np.inf, -np.inf], np.nan)

UOI = ["UOI_score", "link_node_ratio", "connected_node_ratio",
       "intersection_density", "median_block_length_ft",
       "walking_circuity", "pedshed_reach"]
# outcome: (column, friendly label, "good" direction higher/lower)
OUT_ALL = [
    ("mobility_kfr_p25",        "Econ. mobility (kid inc. rank, p25)", "higher"),
    ("incarceration_p25",       "Incarceration rate (p25)",            "lower"),
    ("eviction_filing_rate",    "Eviction filing rate",                "lower"),
    ("ped_fatal_per_100k_pop_yr", "Pedestrian fatalities /100k pop/yr","lower"),
    ("ped_fatal_per_100km2_yr", "Pedestrian fatalities /100km^2/yr",   "lower"),
    ("stable_job_share",        "Stable-job share (CE03/C000)",        "higher"),
    ("job_density_per_sqkm",    "Job density /km^2",                    "higher"),
    ("pct_bachelor_plus",       "Bachelor's degree+ share (25+)",      "higher"),
    ("pct_white",               "White-alone share",                   "n/a"),
    ("pct_black",               "Black-alone share",                   "n/a"),
    ("pct_hispanic",            "Hispanic/Latino share",               "n/a"),
]
OUT = [o for o in OUT_ALL if o[0] in p.columns]   # only outcomes present in the panel

# ---------------- Spearman matrix ----------------
rho = pd.DataFrame(index=UOI, columns=[o[0] for o in OUT], dtype=float)
nmat = pd.DataFrame(index=UOI, columns=[o[0] for o in OUT], dtype=float)
for u in UOI:
    for col, _, _ in OUT:
        d = p[[u, col]].dropna()
        nmat.loc[u, col] = len(d)
        if len(d) > 30:
            rho.loc[u, col] = stats.spearmanr(d[u], d[col]).statistic
rho.to_csv(RES / "corr_matrix_spearman.csv")
nmat.to_csv(RES / "corr_matrix_n.csv")
print("Spearman rho (UOI x outcomes):"); print(rho.round(3).to_string())

# ---------------- heatmap ----------------
labels = [o[1] for o in OUT]
fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="white")
M = rho.values.astype(float)
im = ax.imshow(M, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
ax.set_xticks(range(len(OUT))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(UOI))); ax.set_yticklabels(UOI, fontsize=9)
for i in range(len(UOI)):
    for j in range(len(OUT)):
        v = M[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.35 else "black", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.025, label="Spearman rho")
ax.set_title("UOI vs socioeconomic outcomes — Spearman correlation (US census tracts)")
fig.tight_layout(); fig.savefig(RES / "fig_corr_heatmap.png", dpi=140); plt.close(fig)
print(f"saved {RES/'fig_corr_heatmap.png'}")

# ---------------- scatter panel: UOI_score vs each outcome ----------------
ncol = 3
nrow = int(np.ceil(len(OUT) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5.3 * ncol, 4.5 * nrow), facecolor="white")
axes = np.atleast_1d(axes).ravel()
for ax in axes[len(OUT):]:
    ax.axis("off")
for ax, (col, lab, _) in zip(axes, OUT):
    d = p[["UOI_score", col]].dropna()
    if col in ("eviction_filing_rate", "ped_fatal_per_100km2_yr", "job_density_per_sqkm"):
        d = d[d[col] <= d[col].quantile(0.99)]          # clip extreme tail for display
    hb = ax.hexbin(d["UOI_score"], d[col], gridsize=45, cmap="viridis", mincnt=1, bins="log")
    # binned median trend
    q = pd.qcut(d["UOI_score"], 20, duplicates="drop")
    tr = d.groupby(q, observed=True).median()
    ax.plot(tr["UOI_score"], tr[col], color="red", lw=2, label="binned median")
    r = stats.spearmanr(p[["UOI_score", col]].dropna().values).statistic
    ax.set_xlabel("UOI composite score"); ax.set_ylabel(lab, fontsize=9)
    ax.set_title(f"{lab}\nSpearman rho={r:.3f}  (n={int(nmat.loc['UOI_score', col])})", fontsize=10)
    ax.legend(fontsize=8)
fig.suptitle("UOI composite score vs socioeconomic outcomes (US census tracts)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(RES / "fig_scatter_uoi_vs_outcomes.png", dpi=140); plt.close(fig)
print(f"saved {RES/'fig_scatter_uoi_vs_outcomes.png'}")
