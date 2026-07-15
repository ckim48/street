"""Regime-1 search diagnostics: convergence, acceptance, seed stability.

Regime 1 is a simulated-annealing optimizer (find the cost-efficient network),
NOT a posterior sampler, so R-hat / ESS don't apply directly.  The relevant
"accuracy" evidence for an annealing search is:

  1. CONVERGENCE   -- does the running-best objective J reach a plateau?
  2. ACCEPTANCE    -- does the Metropolis acceptance rate fall smoothly from
                      ~high (exploration, hot) to ~low (exploitation, cold)?
                      Flat-high = random walk (not optimizing); flat-zero =
                      frozen (stuck).  A healthy anneal decays in between.
  3. STABILITY     -- do independent seeds converge to the SAME optimum
                      (access_gain, #added, #removed)?  Tight spread => the
                      search reliably finds the optimum, not a seed artefact.

Runs `optimize` for several seeds, plots all three, writes a diagnostics CSV.

Outputs: results/pathfinder/fig_{slug}_regime1_mcmc.png,
         results/pathfinder/regime1_mcmc_diag.csv
Usage: python pathfinder/22_regime1_diagnostics.py [--slug detroit]
       [--iters 2000] [--seeds 1 7 13 21 42]
"""
from __future__ import annotations

import argparse
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import importlib.util
from pathlib import Path

from pf_common import CITIES, RES

# import the numbered Regime-1 module (name starts with a digit)
_spec = importlib.util.spec_from_file_location(
    "regime1", Path(__file__).with_name("20_regime1_prehighway.py"))
regime1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(regime1)

warnings.filterwarnings("ignore")


def rolling(x, w):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[w:] - c[:-w]) / w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="detroit")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--landmarks", type=int, default=28)
    ap.add_argument("--mu", type=float, default=0.08)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 7, 13, 21, 42])
    a = ap.parse_args()
    cfg = CITIES[a.slug]

    runs, diag_rows = [], []
    for sd in a.seeds:
        print(f"[{a.slug}] seed={sd} ...", flush=True)
        row, art = regime1.optimize(a.slug, a.iters, a.landmarks, a.mu, sd)
        tr = np.array(art["trace"], float)   # it,T,J,bestJ,accepted,n_edges
        runs.append((sd, tr))
        diag_rows.append(dict(
            seed=sd, iters=a.iters,
            access_gain=row["access_gain"], eff_gain=row["eff_gain"],
            n_added=row["n_added"], n_removed=row["n_removed"],
            len_opt_km=row["len_opt_km"],
            final_bestJ=round(float(tr[-1, 3]), 5),
            accept_rate=round(float(tr[:, 4].mean()), 3),
            accept_rate_last10pct=round(float(tr[int(0.9 * len(tr)):, 4].mean()), 3),
        ))
        print(f"    access_gain={row['access_gain']*100:.2f}%  "
              f"+{row['n_added']}/-{row['n_removed']}  "
              f"accept={diag_rows[-1]['accept_rate']:.2f}", flush=True)

    df = pd.DataFrame(diag_rows)
    df.to_csv(RES / "regime1_mcmc_diag.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(runs)))

    # (1) convergence: running-best J
    ax = axes[0]
    for (sd, tr), c in zip(runs, cmap):
        ax.plot(tr[:, 0], tr[:, 3] * 100, color=c, lw=1.3, label=f"seed {sd}")
    ax.set_xlabel("iteration"); ax.set_ylabel("running-best objective J (%)")
    ax.set_title("(1) Convergence of best J", fontsize=11)
    ax.legend(fontsize=8)

    # (2) acceptance rate (rolling) vs iteration, with temperature on twin axis
    ax = axes[1]
    W = max(20, a.iters // 40)
    for (sd, tr), c in zip(runs, cmap):
        ar = rolling(tr[:, 4], W)
        ax.plot(np.arange(len(ar)), ar, color=c, lw=1.2)
    ax.set_xlabel("iteration"); ax.set_ylabel(f"acceptance rate (roll {W})")
    ax.set_ylim(0, 1); ax.set_title("(2) Acceptance rate (anneal)", fontsize=11)
    axT = ax.twinx()
    axT.plot(runs[0][1][:, 0], runs[0][1][:, 1], color="#d73027", ls="--", lw=1)
    axT.set_ylabel("temperature", color="#d73027")
    axT.tick_params(axis="y", labelcolor="#d73027")

    # (3) seed stability of the optimum
    ax = axes[2]
    ax.scatter(df["seed"].astype(str), df["access_gain"] * 100,
               s=70, color="#2166ac", zorder=3)
    m, sdv = df["access_gain"].mean() * 100, df["access_gain"].std() * 100
    ax.axhline(m, color="#888", ls="--", lw=1)
    ax.fill_between([-0.4, len(df) - 0.6], m - sdv, m + sdv, color="#2166ac", alpha=0.12)
    ax.set_xlabel("seed"); ax.set_ylabel("final access_gain (%)")
    ax.set_title(f"(3) Seed stability: {m:.2f}% ± {sdv:.2f}%", fontsize=11)

    fig.suptitle(f"{cfg['city']} Regime-1 search diagnostics "
                 f"({len(runs)} seeds × {a.iters} iters)", fontsize=12)
    fig.tight_layout()
    fig.savefig(RES / f"fig_{a.slug}_regime1_mcmc.png", dpi=130)
    plt.close(fig)

    print("\nwrote regime1_mcmc_diag.csv, fig_"
          f"{a.slug}_regime1_mcmc.png")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
