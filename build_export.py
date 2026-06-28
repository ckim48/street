#!/usr/bin/env python3
"""Consolidate every stage's figures + data into a single export bundle,
and compute national headline statistics. Run in the `street` env."""
import json, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/wnlab/CK_street")
STAMP = "2026-06-28"
EXP = ROOT / "results" / f"export_{STAMP}"
(EXP / "01_national_uoi").mkdir(parents=True, exist_ok=True)
(EXP / "02_top1000").mkdir(parents=True, exist_ok=True)
(EXP / "03_mcmc_optimal").mkdir(parents=True, exist_ok=True)
(EXP / "04_gnn_surrogate").mkdir(parents=True, exist_ok=True)
(EXP / "05_virtual_synthesis").mkdir(parents=True, exist_ok=True)
(EXP / "06_alabama_casestudy").mkdir(parents=True, exist_ok=True)

def cp(src, dstdir):
    src = ROOT / src
    if src.exists():
        shutil.copy2(src, EXP / dstdir / src.name)
        return True
    return False

stats = {}

# ---------- Stage 1+2: national spec metrics ----------
m = pd.read_parquet(ROOT / "data/outputs/uoi_spec_metrics.parquet")
m["GEOID"] = m["GEOID"].astype(str).str.zfill(11)
m["state"] = m["GEOID"].str[:2]
METRICS = ["link_node_ratio", "connected_node_ratio", "intersection_density",
           "median_block_length_ft", "walking_circuity", "pedshed_reach"]
OKFLAGS = ["lnr_ok", "cnr_ok", "inter_density_ok", "block_ok", "circuity_ok"]
ok = m[m["status"].astype(str).str.lower().eq("ok")] if "status" in m else m
stats["national"] = {
    "tracts_total": int(len(m)),
    "tracts_with_metrics": int(m[METRICS].notna().all(axis=1).sum()),
    "states_covered": int(m["state"].nunique()),
    "metric_medians": {k: round(float(m[k].median()), 4) for k in METRICS},
    "metric_means": {k: round(float(m[k].mean()), 4) for k in METRICS},
    "ok_flag_rates": {k: round(float(m[k].mean()), 4) for k in OKFLAGS if k in m},
}
m.to_parquet(EXP / "01_national_uoi" / "uoi_spec_metrics_national.parquet", index=False)
m.to_csv(EXP / "01_national_uoi" / "uoi_spec_metrics_national.csv", index=False)

# per-state coverage + median table
per_state = (m.groupby("state")
               .agg(tracts=("GEOID", "size"),
                    lnr=("link_node_ratio", "median"),
                    cnr=("connected_node_ratio", "median"),
                    dens=("intersection_density", "median"),
                    block_ft=("median_block_length_ft", "median"),
                    circuity=("walking_circuity", "median"),
                    pedshed=("pedshed_reach", "median"))
               .round(4).reset_index().sort_values("tracts", ascending=False))
per_state.to_csv(EXP / "01_national_uoi" / "per_state_summary.csv", index=False)

# national distribution figure: 6 metric histograms with doc bounds
BOUNDS = {  # (low, high, direction text)
    "link_node_ratio": (1.4, None, ">=1.4"),
    "connected_node_ratio": (0.7, None, ">=0.7"),
    "intersection_density": (140, None, ">140"),
    "median_block_length_ft": (None, 600, "<=600"),
    "walking_circuity": (1.2, 1.7, "1.2-1.7"),
    "pedshed_reach": (None, None, "higher"),
}
fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="white")
for ax, k in zip(axes.ravel(), METRICS):
    v = m[k].dropna()
    lo, hi = v.quantile(0.01), v.quantile(0.99)
    ax.hist(v.clip(lo, hi), bins=60, color="#4C72B0", alpha=0.85)
    b = BOUNDS[k]
    if b[0] is not None:
        ax.axvline(b[0], color="crimson", ls="--", lw=1.4)
    if b[1] is not None:
        ax.axvline(b[1], color="crimson", ls="--", lw=1.4)
    ax.set_title(f"{k}\n(doc: {b[2]}, median={v.median():.3g})", fontsize=10)
    ax.set_yticks([])
fig.suptitle(f"National UOI spec-metric distributions — {len(m):,} tracts, "
             f"{m['state'].nunique()} states", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(EXP / "01_national_uoi" / "fig_national_metric_distributions.png", dpi=140)
plt.close(fig)

# existing national figures
cp("results/figures/uoi_correlation.png", "01_national_uoi")
cp("results/figures/uoi_maps.png", "01_national_uoi")
cp("results/uoi_spec_vs_code.csv", "01_national_uoi")

# ---------- Stage 3: top-1000 ----------
for f in ["fig_top24_networks.png", "fig_mid24_networks.png", "fig_last24_networks.png",
          "fig_score_distribution.png", "fig_metric_profile.png",
          "fig_metric_correlation.png", "fig_top1000_by_state.png",
          "top1000_uoi.csv", "uoi_scores_all.csv"]:
    cp(f"results/top1000/{f}", "02_top1000")

# ---------- Stage 5: MCMC optimal ----------
for f in ["dtf_table.csv", "fig_dtf_distribution.png", "fig_metric_shift.png",
          "fig_best_networks.png", "fig_optimal_gallery.png"]:
    cp(f"results/mcmc_spec/{f}", "03_mcmc_optimal")
net_dir = EXP / "03_mcmc_optimal" / "per_tract_networks"
net_dir.mkdir(exist_ok=True)
for p in sorted((ROOT / "results/mcmc_spec/networks").glob("*.png")):
    shutil.copy2(p, net_dir / p.name)

# dtf stats from summary.json (now 1800)
summ = json.loads((ROOT / "data/outputs/sampler_spec/summary.json").read_text())
dtf = np.array([v["distance_to_frontier"] for v in summ.values()])
top1000_ids = set(pd.read_csv(ROOT / "results/top1000/top1000_uoi.csv")["GEOID"].astype(str).str.zfill(11))
in_top = np.array([g.zfill(11) in top1000_ids for g in summ])
stats["mcmc"] = {
    "tracts_total": int(len(dtf)),
    "top1000_subset": int(in_top.sum()),
    "national_sample_subset": int((~in_top).sum()),
    "dtf_median": round(float(np.median(dtf)), 4),
    "dtf_mean": round(float(np.mean(dtf)), 4),
    "dtf_p90": round(float(np.percentile(dtf, 90)), 4),
    "near_optimal_frac_lt0.05": round(float((dtf < 0.05).mean()), 4),
    "top1000_dtf_median": round(float(np.median(dtf[in_top])), 4) if in_top.any() else None,
    "national_dtf_median": round(float(np.median(dtf[~in_top])), 4) if (~in_top).any() else None,
}
# dtf: top1000 vs national-sample comparison fig
fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
ax.hist(dtf[in_top], bins=40, alpha=0.6, label=f"top-1000 elite (n={in_top.sum()})", color="#4C72B0")
ax.hist(dtf[~in_top], bins=40, alpha=0.6, label=f"national stratified (n={(~in_top).sum()})", color="#DD8452")
ax.set_xlabel("distance to frontier (dtf)"); ax.set_ylabel("tracts")
ax.set_title("MCMC optimality gap: elite vs national sample")
ax.legend()
fig.tight_layout()
fig.savefig(EXP / "03_mcmc_optimal" / "fig_dtf_elite_vs_national.png", dpi=140)
plt.close(fig)

# ---------- Stage 6: GNN ----------
cp("results/gnn/fig_pred_vs_true.png", "04_gnn_surrogate")
cp("results/gnn/fig_pred_vs_true_top1000_bak.png", "04_gnn_surrogate")
cp("data/outputs/gnn_dtf_predictions.csv", "04_gnn_surrogate")

# ---------- Stage 5b: synthesis ----------
cp("results/synth/fig_synth_networks.png", "05_virtual_synthesis")
cp("results/synth/synth_metrics.csv", "05_virtual_synthesis")
cp("results/synth/run.log", "05_virtual_synthesis")

# ---------- Alabama case study ----------
for f in ["county_01073_birmingham.png", "network_uoi_01033.png", "network_uoi_01081.png",
          "network_uoi_01115.png", "uoi_correlation_01.png", "uoi_maps_01.png"]:
    cp(f"results/state_01/figures/{f}", "06_alabama_casestudy")
for f in ["network_counties_uoi.csv", "uoi_metrics_01.csv", "uoi_summary_01.csv",
          "uoi_verification_01.csv"]:
    cp(f"results/state_01/tables/{f}", "06_alabama_casestudy")

# ---------- manifest ----------
(EXP / "SUMMARY_STATS.json").write_text(json.dumps(stats, indent=2))
files = sorted(p.relative_to(EXP).as_posix() for p in EXP.rglob("*") if p.is_file())
print(json.dumps(stats, indent=2))
print(f"\n{len(files)} files in {EXP}")
print("\n".join("  " + f for f in files))
