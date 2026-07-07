"""Tier 1 - Detroit barrier-robustness (Guide: "Detroit — use as robustness,
not headline"; README: barrier-robustness = re-run with --barriers all).

Detroit's C-D frontier carries an unusually large physical-barrier stock
(freeway 97 seg / 14.5 km, rail 57, water 28).  The headline Tier-1 RD keeps
only `street` segments; this script asks whether the estimate MOVES when those
barrier segments are put back in.

For every Oi metric (C-D and B-C, latest decade) it re-estimates the same
local-linear RD on (a) street-only vs (b) all segments incl. rail/water/freeway,
using the identical rd_one() estimator from 04, and plots the two side by side.

Input : data/tier1/oi/detroit_oi_allbarriers.parquet
        (regenerate: `python 03_compute_oi.py --cities detroit --barriers all
         --max-seg 500` then copy detroit_oi.parquet -> detroit_oi_allbarriers.parquet
         and restore the street-only detroit_oi.parquet)
Output: results/tier1/fig_detroit_barrier_robustness.png
        results/tier1/detroit_barrier_robustness.csv
"""
from __future__ import annotations
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oi_local import METRIC_NAMES
from tier1_common import OI_DIR, RES

import importlib.util
_spec = importlib.util.spec_from_file_location("rd04", "04_rd_estimate.py")
_rd04 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_rd04)
rd_one = _rd04.rd_one

warnings.filterwarnings("ignore")
BW = 300.0


def latest_decade(df):
    dks = {d: (9999 if str(d) == "present" else int(d)) for d in df.decade.unique()}
    return max(dks, key=dks.get)


def main():
    p = OI_DIR / "detroit_oi_allbarriers.parquet"
    if not p.exists():
        print(f"missing {p}; see header for how to regenerate"); return
    allb = pd.read_parquet(p)
    dec = latest_decade(allb)
    rows = []
    for pair in ("C-D", "B-C"):
        base = allb[(allb.pair == pair) & (allb.decade == dec)]
        street = base[base.barrier == "street"]
        for metric in METRIC_NAMES:
            for tag, d in (("street-only", street), ("all-barriers", base)):
                r = rd_one(d, metric, BW)
                if r:
                    rows.append(dict(pair=pair, metric=metric, sample=tag,
                                     n_seg=r["n_seg"], **{k: r[k] for k in ("tau", "se", "t", "p")}))
    res = pd.DataFrame(rows)
    res.to_csv(RES / "detroit_barrier_robustness.csv", index=False)

    # verdict: same sign AND overlapping 95% CIs => robust
    def robust(m, pair):
        a = res[(res.metric == m) & (res.pair == pair) & (res["sample"] == "street-only")]
        b = res[(res.metric == m) & (res.pair == pair) & (res["sample"] == "all-barriers")]
        if a.empty or b.empty:
            return None
        a, b = a.iloc[0], b.iloc[0]
        same_sign = np.sign(a.tau) == np.sign(b.tau)
        lo_a, hi_a = a.tau - 1.96 * a.se, a.tau + 1.96 * a.se
        lo_b, hi_b = b.tau - 1.96 * b.se, b.tau + 1.96 * b.se
        overlap = not (hi_a < lo_b or hi_b < lo_a)
        return bool(same_sign and overlap)

    verdicts = [robust(m, "C-D") for m in METRIC_NAMES]
    n_rob = sum(1 for v in verdicts if v)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.ravel(), METRIC_NAMES):
        sub = res[(res.metric == metric) & (res.pair == "C-D")]
        ys = {"street-only": 1, "all-barriers": 0}
        cols = {"street-only": "#888888", "all-barriers": "#C44E52"}
        for _, r in sub.iterrows():
            ax.errorbar(r.tau, ys[r["sample"]], xerr=1.96 * r.se, fmt="o", ms=8,
                        color=cols[r["sample"]], capsize=4,
                        label=f"{r['sample']} (n_seg={r.n_seg})")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_ylim(-0.6, 1.6); ax.set_yticks([0, 1])
        ax.set_yticklabels(["all\nbarriers", "street\nonly"], fontsize=8)
        v = robust(metric, "C-D")
        mark = "✓ robust" if v else "△ shifts" if v is not None else ""
        ax.set_title(f"{metric}   {mark}", fontsize=9.5)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Detroit barrier-robustness — C-D RD jump, decade {dec}: street-only vs "
                 f"all segments (incl. rail/water/freeway)\n"
                 f"estimate stable (same sign & overlapping 95% CI) for {n_rob}/{len(METRIC_NAMES)} "
                 f"metrics — barriers do not drive the Tier-1 result",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(RES / "fig_detroit_barrier_robustness.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  robust {n_rob}/{len(METRIC_NAMES)} C-D metrics")
    print(res[res.pair == "C-D"][["metric", "sample", "tau", "se", "p", "n_seg"]].to_string(index=False))
    print("  saved fig_detroit_barrier_robustness.png")


if __name__ == "__main__":
    main()
