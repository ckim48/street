"""Tier 1 - CHRONEX year-field robustness (M1 baseline vs M2).

The headline pipeline dates roads with `yr_upper_M1`, under which the historic
HOLC core reads as "built by 1940", so decade graphs barely change and the
Oi-by-decade trajectories are flat.  The guide/README ask for a robustness pass
with a younger year-field.  This rebuilds the whole 02->03->04 chain with
`yr_upper_M2` into SEPARATE _m2 dirs (M1 outputs untouched) and plots the C-D
intersection-density RD trajectory M1 vs M2 for each city.

CAVEAT (honest): ~46% of CHRONEX segments have no valid M2 year and are dropped,
so M2 networks are younger BUT more disconnected — any extra "divergence" under
M2 is partly a missing-data artifact, not only real construction timing.  The
robustness question is whether the RD *sign/direction* survives the year-field
swap, not whether M2 trajectories are the truth.

Outputs: data/tier1/{graphs_m2,oi_m2,rd_m2}/ , results/tier1/rd_m2_all.csv ,
         results/tier1/fig_tier1_m2_robustness.png
Usage: python tier1/08_m2_robustness.py            # all 6 cities
"""
from __future__ import annotations
import importlib.util
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tier1_common as tc
from tier1_common import CITIES, DECADES, RES

warnings.filterwarnings("ignore")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    # derive _m2 dirs from the real (absolute) tier1_common paths so cwd is irrelevant
    G2 = tc.GRAPH_DIR.parent / "graphs_m2"
    O2 = tc.OI_DIR.parent / "oi_m2"
    R2 = tc.RD_DIR.parent / "rd_m2"
    for d in (G2, O2, R2):
        d.mkdir(parents=True, exist_ok=True)

    m02 = _load("m02", "02_decade_graphs.py")
    m03 = _load("m03", "03_compute_oi.py")
    m04 = _load("m04", "04_rd_estimate.py")
    # redirect every path the imported modules bound from tier1_common (non-destructive)
    m02.GRAPH_DIR = G2
    m03.GRAPH_DIR = G2; m03.OI_DIR = O2
    m04.OI_DIR = O2; m04.RD_DIR = R2

    slugs = list(CITIES)
    YF = "yr_upper_M2"

    print("=== 1) M2 decade graphs ===", flush=True)
    for slug in slugs:
        print(f"[{slug}] {CITIES[slug]['city']}", flush=True)
        m02.build_from_chronex(slug, YF)

    print("=== 2) M2 Oi ===", flush=True)
    for slug in slugs:
        try:
            m03.process_city(slug, "75,150,225,300", 250.0, "street", 300, "")
        except Exception as e:                                  # noqa: BLE001
            print(f"[{slug}] oi FAILED: {type(e).__name__}: {e}", flush=True)

    print("=== 3) M2 RD ===", flush=True)
    allc = []
    for slug in slugs:
        try:
            o = m04.process_city(slug, m04.METRIC_NAMES, 300.0)
            if o is not None:
                allc.append(o)
        except Exception as e:                                  # noqa: BLE001
            print(f"[{slug}] rd FAILED: {type(e).__name__}: {e}", flush=True)
    if not allc:
        print("no M2 RD produced"); return
    m2 = pd.concat(allc, ignore_index=True)
    m2.to_csv(RES / "rd_m2_all.csv", index=False)

    # ---- compare M1 (committed) vs M2 : C-D intersection_density by decade ----
    m1 = pd.read_csv(RES / "rd_all.csv")
    metric = "intersection_density"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, slug in zip(axes.ravel(), slugs):
        for df, col, lab in [(m1, "#888888", "M1 (baseline)"), (m2, "#C44E52", "M2 (younger)")]:
            d = df[(df.slug == slug) & (df.pair == "C-D") & (df.metric == metric)]
            d = d[d.decade.apply(lambda x: str(x).isdigit())].copy()
            d["decade"] = d.decade.astype(int)
            d = d.sort_values("decade")
            if len(d):
                ax.errorbar(d.decade, d.tau, yerr=1.96 * d.se, marker="o", ms=4,
                            color=col, capsize=2, lw=1.4, label=lab)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        # divergence = spread of tau across decades
        def spread(df):
            d = df[(df.slug == slug) & (df.pair == "C-D") & (df.metric == metric)]
            return float(d.tau.max() - d.tau.min()) if len(d) else np.nan
        ax.set_title(f"{CITIES[slug]['city']}\ndecade-spread τ:  M1={spread(m1):.2f}  M2={spread(m2):.2f}",
                     fontsize=9.5)
        ax.set_xlabel("decade"); ax.legend(fontsize=7)
    axes[0][0].set_ylabel("C-D RD jump τ (intersection_density)")
    fig.suptitle("Tier 1 year-field robustness — C-D intersection-density RD by decade: "
                 "M1 baseline vs M2 (younger, ~46% segments dropped → more disconnected)\n"
                 "robustness = does the sign survive; M2 adds temporal spread but is partly a "
                 "missing-data artifact, so M1 stays the headline",
                 fontsize=11.5, y=1.0)
    fig.tight_layout()
    fig.savefig(RES / "fig_tier1_m2_robustness.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_tier1_m2_robustness.png ; rd_m2_all.csv ({len(m2)} rows)")


if __name__ == "__main__":
    main()
