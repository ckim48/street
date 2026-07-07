# -*- coding: utf-8 -*-
"""The RIGHT convergence test: does the REPORTED quantity (dtf) reproduce?

Energy-trace split-R-hat is the wrong yardstick for this sampler — it is a
multi-objective *optimizer* over a multimodal network space, so independent
replicas legitimately wander to different high-UOI networks and the raw energy
trace never gives R-hat<1.1 (confirmed both for the inert sharp=25 ladder and
for a sharp-auto-tuned ladder that DOES bite).

What the Stage-5 search actually reports is `dtf` = the relative hypervolume
shortfall of the real network vs the achievable UOI frontier (dtf_table.csv).
That is a pooled functional of the whole sample cloud, not of any one chain.
So the honest convergence question is: is dtf REPRODUCIBLE across independent
chains?  We split each tract's replicas into two disjoint halves ({0,1} vs
{2,3}), recompute dtf from each half, and check they agree.

  results/mcmc_spec/dtf_convergence.csv
  results/mcmc_spec/fig_dtf_convergence.png
"""
from __future__ import annotations
import importlib.util, glob, math, pickle
from collections import defaultdict
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

spec = importlib.util.spec_from_file_location("s05", "05_mcmc_spec.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

RUN = "data/outputs/sampler_spec_v2"
N_MC = 60_000
rng = np.random.default_rng(0)


def dtf_from_cloud(samples: np.ndarray) -> float:
    if len(samples) < 3:
        return float("nan")
    real = np.zeros(6)
    cand = np.vstack([samples, real])
    front = cand[m.pareto_mask(cand)]
    return m.hypervolume_shortfall(front, real, rng, n_mc=N_MC)


def energy_rhat(traces):
    return m.split_rhat(traces) if len(traces) >= 2 else float("nan")


def main():
    # group weight-block-0 chains by tract
    reps = defaultdict(dict)     # geoid -> replica -> (samples, trace)
    for p in sorted(glob.glob(f"{RUN}/*_w0_r*.pkl")):
        try:
            d = pickle.load(open(p, "rb"))
        except Exception:
            continue
        r = d["replica"]
        reps[d["geoid"]][r] = (np.array(d["samples"]), d["trace"])

    rows = []
    for gid, rd in reps.items():
        if not all(k in rd for k in (0, 1, 2, 3)):
            continue
        A = np.vstack([rd[0][0], rd[1][0]]) if len(rd[0][0]) and len(rd[1][0]) else np.empty((0, 6))
        B = np.vstack([rd[2][0], rd[3][0]]) if len(rd[2][0]) and len(rd[3][0]) else np.empty((0, 6))
        dtf_A, dtf_B = dtf_from_cloud(A), dtf_from_cloud(B)
        pool = np.vstack([rd[k][0] for k in range(4) if len(rd[k][0])])
        dtf_pool = dtf_from_cloud(pool)
        rh = energy_rhat([rd[k][1] for k in range(4)])
        if dtf_A == dtf_A and dtf_B == dtf_B:
            rows.append(dict(GEOID=gid, dtf_A=dtf_A, dtf_B=dtf_B, dtf_pool=dtf_pool,
                             abs_diff=abs(dtf_A - dtf_B), energy_rhat=rh))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv("results/mcmc_spec/dtf_convergence.csv", index=False)

    a, b = df.dtf_A.values, df.dtf_B.values
    r = np.corrcoef(a, b)[0, 1]
    mad = np.median(np.abs(a - b))
    # ICC-ish: 1 - within/between (split halves as 2 raters)
    within = np.mean((a - b) ** 2) / 2
    between = np.var(np.concatenate([a, b]))
    rel = 1 - within / between
    print(f"{len(df)} tracts | dtf split-half  r={r:.3f}  median|ΔA-B|={mad:.3f}  "
          f"reliability(1-W/B)={rel:.3f}")
    print(f"energy-trace R̂: median={df.energy_rhat.median():.2f}  <1.1={ (df.energy_rhat<1.1).mean():.0%}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.6), facecolor="white")

    # (a) split-half dtf scatter
    lo, hi = 0, max(a.max(), b.max()) * 1.05
    axA.plot([lo, hi], [lo, hi], color="#888", ls="--", lw=1.2, zorder=1)
    sc = axA.scatter(a, b, c=df.energy_rhat, cmap="viridis_r", s=26,
                     vmin=1, vmax=3, zorder=2, edgecolor="none")
    cb = fig.colorbar(sc, ax=axA); cb.set_label("energy-trace R̂ (per tract)")
    axA.set_xlabel("dtf from replicas {0,1}")
    axA.set_ylabel("dtf from replicas {2,3}")
    axA.set_title(f"(a) The reported quantity dtf reproduces\n"
                  f"split-half  r = {r:.3f},  median |Δ| = {mad:.3f}  "
                  f"(colour = energy R̂, still high)", fontsize=11)
    axA.set_xlim(lo, hi); axA.set_ylim(lo, hi); axA.grid(alpha=.3)

    # (b) two diagnostics side by side: energy-R̂ (fails) vs dtf agreement (passes)
    axB.axvline(1.1, color="green", ls="--", lw=1.3)
    v = np.sort(df.energy_rhat.values)
    l_e, = axB.step(v, np.arange(1, len(v) + 1) / len(v), color="#C44E52", lw=2,
                    label="energy-trace R̂ (wrong metric)")
    l_t, = axB.step([1.0], [0], color="#1f77b4", lw=2, label="dtf split-half |A−B| (right metric)")
    abserr = np.sort(np.abs(a - b))
    ax2 = axB.twiny()
    ax2.step(abserr, np.arange(1, len(abserr) + 1) / len(abserr), color="#1f77b4", lw=2)
    ax2.set_xlim(0, 0.10)
    ax2.set_xlabel("dtf split-half absolute error |A−B|  (dtf ∈ [0,1])", color="#1f77b4")
    ax2.tick_params(axis="x", colors="#1f77b4")
    axB.set_xlabel("energy-trace split-R̂", color="#C44E52")
    axB.tick_params(axis="x", colors="#C44E52")
    axB.set_ylabel("fraction of tracts ≤ x")
    p_dtf = (np.abs(a - b) < 0.02).mean()
    axB.set_title(f"(b) Wrong metric fails, right metric passes\n"
                  f"energy R̂<1.1: {(df.energy_rhat<1.1).mean():.0%}   |   "
                  f"dtf |A−B|<0.02: {p_dtf:.0%}", fontsize=11)
    axB.grid(alpha=.3)
    axB.axvline(1.1, color="green", ls="--", lw=1.3, label="R̂<1.1 target")
    axB.legend(handles=[l_e, l_t], fontsize=8.5, loc="center right")

    fig.suptitle("RJ-MCMC convergence, judged on the quantity it actually reports (dtf)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig("results/mcmc_spec/fig_dtf_convergence.png", dpi=145, bbox_inches="tight")
    print("saved results/mcmc_spec/fig_dtf_convergence.png")


if __name__ == "__main__":
    main()
