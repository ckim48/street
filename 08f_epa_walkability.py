"""Stage 8f: UOI vs EPA National Walkability Index (independent benchmark).

The EPA Smart Location Database's National Walkability Index (NWI, 1-20) ranks
census BLOCK GROUPS on density, land-use diversity, and transit proximity.  We
aggregate it to census tracts (population-weighted) and compare it to our UOI
(composite UOI_score + the six spec metrics).

Honest caveat surfaced by the script: one NWI input, D3B (street intersection
density), is essentially the SAME quantity as our `intersection_density`, so a
positive UOI-NWI correlation is partly MECHANICAL.  We report the D3B overlap
explicitly and also show the correlation of the *non-density* UOI metrics
(circuity, block length, connectivity) against NWI, which is the independent
signal.

Inputs:  data/external/epa/Natl_WI.gdb  (NationalWalkabilityIndex layer)
         data/outputs/tract_panel.parquet  (UOI_score + 6 metrics, tract level)
Outputs: results/epa_walkability/  corr_uoi_vs_nwi.csv, fig_uoi_vs_nwi.png,
         fig_corr_heatmap.png
"""
from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

GDB = "data/external/epa/Natl_WI.gdb"
PANEL = "data/outputs/tract_panel.parquet"
OUT = Path("results/epa_walkability")
OUT.mkdir(parents=True, exist_ok=True)

NWI_COLS = ["NatWalkInd", "D3B_Ranked", "D4A_Ranked", "D2A_Ranked", "D2B_Ranked", "D3B"]
UOI_COLS = ["UOI_score", "link_node_ratio", "connected_node_ratio",
            "intersection_density", "median_block_length_ft",
            "walking_circuity", "pedshed_reach"]
NWI_LABEL = {"NatWalkInd": "NWI (composite)", "D3B_Ranked": "D3B intersection-density (rank)",
             "D4A_Ranked": "D4A transit proximity (rank)", "D2A_Ranked": "D2A employment mix (rank)",
             "D2B_Ranked": "D2B emp+HH mix (rank)", "D3B": "D3B intersection density (raw)"}


def load_nwi_by_tract():
    """Read the block-group NWI gdb and population-weight it up to tracts."""
    print("reading EPA NWI gdb ...", flush=True)
    g = gpd.read_file(GDB, layer="NationalWalkabilityIndex",
                      columns=["GEOID20", "TotPop"] + NWI_COLS, ignore_geometry=True)
    g["GEOID"] = g["GEOID20"].astype(str).str.zfill(12).str[:11]
    g["w"] = g["TotPop"].clip(lower=0).fillna(0)
    out = {}
    for col in NWI_COLS:
        g["_wx"] = g[col] * g["w"]
        agg = g.groupby("GEOID").agg(wx=("_wx", "sum"), wsum=("w", "sum"),
                                     mean=(col, "mean"))
        out[col] = np.where(agg["wsum"] > 0, agg["wx"] / agg["wsum"], agg["mean"])
    df = pd.DataFrame(out, index=agg.index).reset_index()
    print(f"  aggregated {len(g)} block groups -> {len(df)} tracts", flush=True)
    return df


def main():
    nwi = load_nwi_by_tract()
    uoi = pd.read_parquet(PANEL)[["GEOID"] + UOI_COLS]
    uoi["GEOID"] = uoi["GEOID"].astype(str).str.zfill(11)
    df = uoi.merge(nwi, on="GEOID", how="inner")
    print(f"joined tracts: {len(df)}", flush=True)

    # ---- Spearman correlation matrix (UOI rows x NWI cols) ------------------
    M = np.full((len(UOI_COLS), len(NWI_COLS)), np.nan)
    for i, u in enumerate(UOI_COLS):
        for j, w in enumerate(NWI_COLS):
            s = df[[u, w]].dropna()
            if len(s) > 100:
                M[i, j] = spearmanr(s[u], s[w]).correlation
    cm = pd.DataFrame(M, index=UOI_COLS, columns=NWI_COLS)
    cm.to_csv(OUT / "corr_uoi_vs_nwi.csv")

    rho_head = cm.loc["UOI_score", "NatWalkInd"]
    rho_dens = cm.loc["intersection_density", "D3B_Ranked"]
    rho_circ = cm.loc["walking_circuity", "NatWalkInd"]
    rho_block = cm.loc["median_block_length_ft", "NatWalkInd"]
    print(f"\nSpearman UOI_score vs NWI          = {rho_head:+.3f}", flush=True)
    print(f"  (mechanical) intersection_density vs D3B = {rho_dens:+.3f}", flush=True)
    print(f"  independent: circuity vs NWI      = {rho_circ:+.3f}", flush=True)
    print(f"  independent: block_length vs NWI  = {rho_block:+.3f}", flush=True)

    # ---- heatmap -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(NWI_COLS)))
    ax.set_xticklabels([NWI_LABEL[c] for c in NWI_COLS], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(UOI_COLS)))
    ax.set_yticklabels(UOI_COLS, fontsize=8)
    for i in range(len(UOI_COLS)):
        for j in range(len(NWI_COLS)):
            if not np.isnan(cm.values[i, j]):
                ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    fig.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    ax.set_title(f"UOI vs EPA National Walkability Index — Spearman ρ  (n={len(df):,} tracts)\n"
                 "note: intersection_density ↔ D3B is the same quantity (mechanical)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_corr_heatmap.png", dpi=140); plt.close(fig)

    # ---- headline scatter + mechanical-overlap panel -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    s = df[["UOI_score", "NatWalkInd"]].dropna()
    hb = axes[0].hexbin(s["UOI_score"], s["NatWalkInd"], gridsize=45, cmap="viridis",
                        bins="log", mincnt=1)
    axes[0].set_xlabel("UOI_score (ours)"); axes[0].set_ylabel("EPA NWI (1-20)")
    axes[0].set_title(f"composite: ρ = {rho_head:+.3f}")
    fig.colorbar(hb, ax=axes[0], label="log tracts")
    s2 = df[["intersection_density", "D3B_Ranked"]].dropna()
    hb2 = axes[1].hexbin(s2["intersection_density"], s2["D3B_Ranked"], gridsize=45,
                         cmap="magma", bins="log", mincnt=1)
    axes[1].set_xlabel("our intersection_density"); axes[1].set_ylabel("EPA D3B (rank)")
    axes[1].set_title(f"mechanical overlap: ρ = {rho_dens:+.3f}")
    fig.colorbar(hb2, ax=axes[1], label="log tracts")
    fig.suptitle("UOI vs EPA National Walkability Index", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "fig_uoi_vs_nwi.png", dpi=140); plt.close(fig)

    print(f"\nsaved -> {OUT}/", flush=True)


if __name__ == "__main__":
    main()
