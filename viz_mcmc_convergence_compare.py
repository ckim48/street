"""Before/after RJ-MCMC convergence comparison for the Stage-5 spec search.

Compares split-R̂ for the *same tracts* under two sampler configs:

  BEFORE  data/outputs/sampler_spec  (rhat via results/mcmc_spec/rhat_summary.csv)
          sharp 60, 4 temps, 2 replicas, 4000 iters
  AFTER   data/outputs/sampler_spec_v2 (rhat recomputed from the chain pkls)
          sharp 25, 8 temps, 4 replicas, 6000 iters, warmer floor (beta_min 0.12)

The "after" R̂ is recomputed straight from the per-chain pkls, so this runs on a
*partial* strengthened run and updates as more tracts finish.  Each tract's R̂ is
the max split-R̂ over its Dirichlet-weight blocks (matching rhat_summary.csv).

  results/mcmc_spec/fig_rhat_before_after.png   dumbbell + ECDF panels
  results/mcmc_spec/rhat_before_after.csv       matched per-tract table

Usage: python viz_mcmc_convergence_compare.py
       python viz_mcmc_convergence_compare.py --after data/outputs/sampler_spec_v2
"""
from __future__ import annotations
import argparse, math, pickle
from collections import defaultdict
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def split_rhat(chains):
    """Split-R̂ over a list of energy traces (second half, split in two)."""
    seqs = []
    for c in chains:
        h = np.asarray(c[len(c) // 2:], dtype=float)
        if len(h) < 4:
            return np.nan
        seqs += [h[: len(h) // 2], h[len(h) // 2:]]
    if len(seqs) < 2:
        return np.nan
    L = min(len(s) for s in seqs)
    if L < 2:
        return np.nan
    arr = np.stack([s[:L] for s in seqs])
    W = arr.var(axis=1, ddof=1).mean()
    B = L * arr.mean(axis=1).var(ddof=1)
    return float(math.sqrt((W * (L - 1) / L + B / L) / W)) if W > 0 else np.nan


def after_rhat(after_dir: Path) -> pd.DataFrame:
    """Recompute per-tract max split-R̂ from the strengthened-run chain pkls."""
    by_tract = defaultdict(lambda: defaultdict(list))   # geoid -> w_idx -> [traces]
    for p in sorted(after_dir.glob("*_w*_r*.pkl")):
        try:
            d = pickle.load(open(p, "rb"))
        except Exception:
            continue
        by_tract[d["geoid"]][d["w_idx"]].append(d["trace"])
    rows = []
    for gid, wd in by_tract.items():
        # need >=2 replicas in a weight block for a meaningful R̂
        rhats = [split_rhat(tr) for tr in wd.values() if len(tr) >= 2]
        rhats = [r for r in rhats if r == r]
        n_chain = sum(len(tr) for tr in wd.values())
        if rhats:
            rows.append(dict(GEOID=gid, rhat_after=max(rhats),
                             n_weight=len(rhats), n_chain=n_chain))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="results/mcmc_spec/rhat_summary.csv")
    ap.add_argument("--after", default="data/outputs/sampler_spec_v2")
    ap.add_argument("--out", default="results/mcmc_spec")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    before = pd.read_csv(args.before, dtype={"GEOID": str})[["GEOID", "rhat_max", "n_nodes"]]
    before = before.rename(columns={"rhat_max": "rhat_before"})
    after = after_rhat(Path(args.after))
    if after.empty:
        print("no completed strengthened-run tracts yet; nothing to plot"); return
    m = before.merge(after, on="GEOID", how="inner").dropna(subset=["rhat_before", "rhat_after"])
    m = m.sort_values("rhat_before", ascending=False).reset_index(drop=True)
    m.to_csv(out / "rhat_before_after.csv", index=False)

    n = len(m)
    med_b, med_a = m.rhat_before.median(), m.rhat_after.median()
    improved = (m.rhat_after < m.rhat_before).mean()
    print(f"{n} matched tracts | median R̂ {med_b:.2f} -> {med_a:.2f} | "
          f"{improved:.0%} improved | <1.1: {(m.rhat_before<1.1).mean():.0%}->"
          f"{(m.rhat_after<1.1).mean():.0%} | <1.2: {(m.rhat_before<1.2).mean():.0%}->"
          f"{(m.rhat_after<1.2).mean():.0%}")

    fig = plt.figure(figsize=(15, 8), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.22)

    # (a) dumbbell before->after (show up to 40 rows spanning the range)
    ax = fig.add_subplot(gs[0, 0])
    show = m if n <= 40 else m.iloc[np.linspace(0, n - 1, 40).round().astype(int)]
    y = np.arange(len(show))[::-1]
    for yi, (_, r) in zip(y, show.iterrows()):
        better = r.rhat_after < r.rhat_before
        ax.plot([r.rhat_before, r.rhat_after], [yi, yi],
                color="#4C72B0" if better else "#C44E52", lw=1.4, zorder=1, alpha=.8)
    ax.scatter(show.rhat_before, y, s=34, color="#999999", zorder=2, label="before (sharp60,temps4,rep2)")
    ax.scatter(show.rhat_after, y, s=34, color="#1f77b4", zorder=3, label="after (sharp25,temps8,rep4,6k)")
    ax.axvline(1.1, color="green", ls="--", lw=1.4, label="target R̂ < 1.1")
    ax.axvline(1.2, color="orange", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xticks([1, 1.1, 1.2, 1.5, 2, 3, 5, 10])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_yticks(y[::max(1, len(y)//20)])
    ax.set_yticklabels(show.GEOID.iloc[::max(1, len(y)//20)].tolist(), fontsize=6)
    ax.set_xlabel("split-R̂ (max over weight blocks, log scale)")
    ax.set_title(f"Strengthened RJ-MCMC convergence — {n} tracts\n"
                 f"median R̂ {med_b:.2f} → {med_a:.2f}  ({improved:.0%} improved)",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower right")

    # (b) ECDF before vs after
    ax = fig.add_subplot(gs[0, 1])
    for col, color, lab in [("rhat_before", "#999999", "before"),
                            ("rhat_after", "#1f77b4", "after")]:
        v = np.sort(m[col].values)
        ax.step(v, np.arange(1, len(v)+1)/len(v), color=color, lw=2, label=lab)
    ax.axvline(1.1, color="green", ls="--", lw=1.4)
    ax.axvline(1.2, color="orange", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xticks([1, 1.1, 1.2, 1.5, 2, 3, 5, 10])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("split-R̂ (log)"); ax.set_ylabel("fraction of tracts ≤ x")
    ax.set_title(f"Convergence CDF — reaching R̂<1.1: "
                 f"{(m.rhat_before<1.1).mean():.0%} → {(m.rhat_after<1.1).mean():.0%}",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=.3)

    fig.suptitle("RJ-MCMC convergence: baseline vs strengthened sampler "
                 f"(same {n} tracts)", fontsize=13, y=1.0)
    fig.savefig(out / "fig_rhat_before_after.png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out/'fig_rhat_before_after.png'}")


if __name__ == "__main__":
    main()
