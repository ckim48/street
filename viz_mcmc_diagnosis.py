# -*- coding: utf-8 -*-
"""Why the strengthened RJ-MCMC still does not reach R-hat<1.1 — the diagnosis.

Two panels, both computed from the strengthened-run chain pkls
(data/outputs/sampler_spec_v2, temps 8, sharp 25, beta_min 0.12):

  (a) Per-temperature acceptance rate along the tempering ladder.  It is FLAT:
      the cold chain (beta=1) and the hottest chain (beta=0.12) accept at the
      same ~0.36 rate.  Each proposal is a small local network edit, so the
      energy change dE is tiny and sharp*beta*dE ~ 0 at every rung -> the
      Metropolis ratio is driven by the proposal geometry (logH), not by the
      energy, so temperature never bites.  The ladder is inert, which is why
      adding rungs (4->8), warming the floor (0.18->0.12) and doubling replicas
      did essentially nothing to split-R-hat.

  (b) Cold-chain energy traces of the 4 replicas of one high-R-hat weight block.
      They plateau at DIFFERENT energy levels: independent replicas random-walk
      into different regions of a multimodal network landscape and stay there.
      Split-R-hat on the energy trace is therefore structurally high -- this is
      genuine multi-modality of the target, not a burn-in / chain-length issue.

  results/mcmc_spec/fig_mcmc_diagnosis.png

Usage: python viz_mcmc_diagnosis.py
"""
from __future__ import annotations
import argparse, math, pickle
from collections import defaultdict
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def split_rhat(chains):
    seqs = []
    for c in chains:
        h = np.asarray(c[len(c) // 2:], dtype=float)
        if len(h) < 4:
            return np.nan
        seqs += [h[: len(h) // 2], h[len(h) // 2:]]
    L = min(len(s) for s in seqs)
    if L < 2:
        return np.nan
    arr = np.stack([s[:L] for s in seqs])
    W = arr.var(axis=1, ddof=1).mean()
    B = L * arr.mean(axis=1).var(ddof=1)
    return float(math.sqrt((W * (L - 1) / L + B / L) / W)) if W > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", default="data/outputs/sampler_spec_v2")
    ap.add_argument("--out", default="results/mcmc_spec")
    ap.add_argument("--n-temps", type=int, default=8)
    ap.add_argument("--beta-min", type=float, default=0.12)
    args = ap.parse_args()
    after = Path(args.after)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    betas = np.geomspace(1.0, args.beta_min, args.n_temps)

    # ---- collect per-temp accept rates + traces grouped by weight block ------
    acc = []                                   # per-temp accept_rate arrays
    blocks = defaultdict(list)                 # (geoid,w_idx) -> [(replica,trace)]
    for p in sorted(after.glob("*_w*_r*.pkl")):
        try:
            d = pickle.load(open(p, "rb"))
        except Exception:
            continue
        ar = d.get("accept_rate")
        if ar and len(ar) == args.n_temps:
            acc.append(ar)
        blocks[(d["geoid"], d["w_idx"])].append((d["replica"], d["trace"]))
    acc = np.array(acc)

    # pick a representative high-R-hat block with 4 replicas and clean traces
    cand = []
    for key, trs in blocks.items():
        if len(trs) >= 4:
            r = split_rhat([t for _, t in trs])
            if r == r:
                cand.append((r, key, trs))
    cand.sort()
    # take one near the 75th percentile of R-hat so it is "typically bad", not the freak
    pick = cand[int(0.75 * (len(cand) - 1))] if cand else None

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.2), facecolor="white")

    # (a) per-temperature acceptance ------------------------------------------
    m = acc.mean(0); lo = np.percentile(acc, 25, 0); hi = np.percentile(acc, 75, 0)
    x = np.arange(args.n_temps)
    axA.fill_between(x, lo, hi, color="#4C72B0", alpha=.18, label="IQR over chains")
    axA.plot(x, m, "-o", color="#1f4e79", lw=2, ms=7, label="mean accept rate")
    axA.set_ylim(0, max(0.6, hi.max() * 1.25))
    axA.set_xticks(x)
    axA.set_xticklabels([f"{b:.2f}" for b in betas])
    axA.set_xlabel(r"tempering rung  (inverse temperature $\beta$;  1.0 = cold, 0.12 = hottest)")
    axA.set_ylabel("Metropolis acceptance rate")
    axA.set_title("(a) The tempering ladder is inert\n"
                  "acceptance is flat cold→hot → temperature never modulates the chain",
                  fontsize=11)
    axA.axhline(m.mean(), color="#C44E52", ls="--", lw=1.2,
                label=f"flat at ~{m.mean():.2f}")
    axA.legend(fontsize=9, loc="upper right"); axA.grid(alpha=.3)
    axA.annotate("hot chain explores no more\nfreely than the cold chain",
                 xy=(x[-1], m[-1]), xytext=(x[-1]-3.4, m.mean()+0.14),
                 fontsize=9, color="#C44E52",
                 arrowprops=dict(arrowstyle="->", color="#C44E52"))

    # (b) multimodal cold-chain traces ----------------------------------------
    if pick:
        rhat, (gid, w_idx), trs = pick
        trs = sorted(trs)
        colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(trs)))
        for (rep, tr), c in zip(trs, colors):
            axB.plot(np.arange(len(tr)), tr, color=c, lw=1.3, alpha=.9,
                     label=f"replica {rep}")
        # plateau levels (2nd-half means)
        for (rep, tr), c in zip(trs, colors):
            lv = np.mean(tr[len(tr)//2:])
            axB.axhline(lv, color=c, ls=":", lw=1, alpha=.7)
        axB.set_xlabel("cold-chain sample (every 10 iters)")
        axB.set_ylabel(r"energy $E=\mathbf{w}\cdot\Delta_{\mathrm{UOI}}$  (cold chain)")
        axB.set_title(f"(b) Replicas settle in different modes\n"
                      f"tract {gid}, weight block {w_idx} — split-R̂ = {rhat:.2f}",
                      fontsize=11)
        axB.legend(fontsize=8, loc="best", ncol=2)
        axB.grid(alpha=.3)

    fig.suptitle("Why the strengthened RJ-MCMC still misses R̂<1.1: an inert "
                 "tempering ladder over a multimodal target",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig_mcmc_diagnosis.png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"per-temp mean accept: {np.round(m,3)}")
    print(f"picked block: {pick[1] if pick else None}  R-hat={pick[0]:.2f}" if pick else "no block")
    print(f"saved {out/'fig_mcmc_diagnosis.png'}")


if __name__ == "__main__":
    main()
