"""RJ-MCMC convergence diagnostics for the Stage-5 spec optimal-network search.

The sampler (05_mcmc_spec.py) runs reversible-jump MCMC with parallel tempering;
each tract is searched by several independent replica chains per Dirichlet weight
draw.  This script turns the saved chain payloads into the standard convergence
panel a reviewer expects:

  fig_mcmc_convergence.png   4-panel diagnostics:
     (a) cold-chain energy traces, several replicas overlaid for a few example
         tracts  -> visual mixing / stationarity check
     (b) split-R-hat distribution across all tracts, with the 1.1 / 1.2 lines
         (the design-doc convergence target is R-hat < 1.1)
     (c) acceptance rate (cold chain) and replica-swap rate distributions
     (d) R-hat vs network size (n_nodes) -> shows where mixing degrades
  rhat_summary.csv           per-tract max R-hat + accept/swap + n_nodes

Inputs : data/outputs/sampler_spec/{summary.json, *.pkl},
         data/outputs/uoi_spec_metrics.parquet (for n_nodes)
Usage  : python viz_mcmc_convergence.py [--examples 4 --replicas 6]
"""
from __future__ import annotations
import argparse, json, pickle
from collections import defaultdict

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uoi_common import OUT_DIR

SAMP = OUT_DIR / "sampler_spec"
RES = OUT_DIR.parent.parent / "results" / "mcmc_spec"
RES.mkdir(parents=True, exist_ok=True)


def load_summary() -> pd.DataFrame:
    s = json.loads((SAMP / "summary.json").read_text())
    rows = []
    for gid, d in s.items():
        rhats = list(d["rhat"].values()) if d.get("rhat") else []
        rows.append({
            "GEOID": gid, "state": gid[:2],
            "rhat_max": max(rhats) if rhats else np.nan,
            "rhat_mean": float(np.mean(rhats)) if rhats else np.nan,
            "accept_cold": d.get("accept_rate_cold", np.nan),
            "swap_rate": d.get("swap_rate", np.nan),
            "dtf": d.get("distance_to_frontier", np.nan),
        })
    df = pd.DataFrame(rows)
    try:
        m = pd.read_parquet(OUT_DIR / "uoi_spec_metrics.parquet")[["GEOID", "n_nodes"]]
        df = df.merge(m, on="GEOID", how="left")
    except Exception:
        df["n_nodes"] = np.nan
    return df


def example_traces(geoid: str, max_rep: int):
    """All replica cold-chain energy traces saved for one tract."""
    out = []
    for p in sorted(SAMP.glob(f"{geoid}_w*_r*.pkl")):
        d = pickle.load(open(p, "rb"))
        out.append(np.asarray(d["trace"], dtype=float))
        if len(out) >= max_rep:
            break
    return out


def pick_examples(df, k):
    """A spread of tracts: best-mixing, median, and worst-mixing by R-hat."""
    d = df.dropna(subset=["rhat_max"]).sort_values("rhat_max")
    if len(d) == 0:
        return []
    idx = np.unique(np.linspace(0, len(d) - 1, k).round().astype(int))
    return d.iloc[idx]["GEOID"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--replicas", type=int, default=8,
                    help="max replica traces to overlay per example tract")
    args = ap.parse_args()

    df = load_summary()
    df.round(4).to_csv(RES / "rhat_summary.csv", index=False)
    n = len(df)
    rh = df["rhat_max"].dropna()
    frac_11 = (rh < 1.1).mean()
    frac_12 = (rh < 1.2).mean()
    print(f"{n} tracts; R-hat median {rh.median():.3f}  "
          f"<1.1: {frac_11:.1%}  <1.2: {frac_12:.1%}")

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    # (a) energy traces, replicas overlaid, for a few example tracts
    ax = fig.add_subplot(gs[0, 0])
    examples = pick_examples(df, args.examples)
    cmap = plt.get_cmap("tab10")
    for ei, gid in enumerate(examples):
        traces = example_traces(gid, args.replicas)
        rhat = df.loc[df.GEOID == gid, "rhat_max"].iloc[0]
        for tr in traces:
            x = np.linspace(0, 1, len(tr))
            ax.plot(x, tr, color=cmap(ei % 10), lw=0.7, alpha=0.55)
        # label once per tract
        ax.plot([], [], color=cmap(ei % 10), lw=1.6,
                label=f"{gid}  R̂={rhat:.2f}")
    ax.set_xlabel("sampling progress (fraction of post-burn-in iterations)")
    ax.set_ylabel("cold-chain energy  E")
    ax.set_title("(a) Energy traces — independent replicas overlaid\n"
                 "(well-mixed ⇒ replicas of a tract overlap & flatten)",
                 fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

    # (b) R-hat distribution
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(rh.clip(upper=3.0), bins=60, color="#4C72B0")
    ax.axvline(1.1, color="green", lw=1.5, ls="--", label="target R̂ < 1.1")
    ax.axvline(1.2, color="orange", lw=1.2, ls=":", label="R̂ = 1.2")
    ax.axvline(rh.median(), color="crimson", lw=1.5,
               label=f"median = {rh.median():.2f}")
    ax.set_xlabel("split-R̂ (max over weight blocks, clipped at 3)")
    ax.set_ylabel("number of tracts")
    ax.set_title(f"(b) Convergence across {n:,} tracts — "
                 f"{frac_11:.0%} reach R̂<1.1", fontsize=10)
    ax.legend(fontsize=8)

    # (c) acceptance & swap rate distributions
    ax = fig.add_subplot(gs[1, 0])
    ac = df["accept_cold"].dropna(); sw = df["swap_rate"].dropna()
    ax.hist(ac, bins=40, color="#55A868", alpha=0.7,
            label=f"cold-chain accept (median {ac.median():.2f})")
    ax.hist(sw, bins=40, color="#C44E52", alpha=0.6,
            label=f"replica swap (median {sw.median():.2f})")
    ax.set_xlabel("rate"); ax.set_ylabel("number of tracts")
    ax.set_title("(c) Move acceptance & parallel-tempering swap rates\n"
                 "(healthy sampler ⇒ moderate accept, frequent swaps)",
                 fontsize=10)
    ax.legend(fontsize=8)

    # (d) R-hat vs network size
    ax = fig.add_subplot(gs[1, 1])
    d2 = df.dropna(subset=["rhat_max", "n_nodes"])
    sc = ax.scatter(d2["n_nodes"], d2["rhat_max"].clip(upper=3.0),
                    s=10, alpha=0.4, c=d2["dtf"], cmap="viridis")
    ax.axhline(1.1, color="green", lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("network size (n_nodes, log)")
    ax.set_ylabel("split-R̂ (clipped at 3)")
    ax.set_title("(d) Mixing vs network size — larger nets mix worse",
                 fontsize=10)
    fig.colorbar(sc, ax=ax, fraction=0.04, label="distance to frontier")

    fig.suptitle("RJ-MCMC convergence diagnostics — Stage-5 spec optimal-network "
                 f"search ({n:,} tracts)", fontsize=13, y=0.98)
    fig.savefig(RES / "fig_mcmc_convergence.png", dpi=145,
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved {RES/'fig_mcmc_convergence.png'}")
    print(f"saved {RES/'rhat_summary.csv'}")


if __name__ == "__main__":
    main()
