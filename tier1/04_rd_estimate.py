"""Tier 1 - Step 5: city-level geographic RD at HOLC grade boundaries.

For each city x boundary-pair x decade x Oi metric we estimate the discontinuity
at the HOLC frontier with a local-linear RD:

    y = a + tau*T + b1*x + b2*(T*x) + (boundary-segment fixed effects)

  * x  = signed distance to the boundary (+ on the lower-grade / treated side);
  * T  = 1{x>0}, the lower-grade side treatment (Guide Step 5.2);
  * triangular kernel weights K(x/h) inside bandwidth h;
  * boundary-segment fixed effects absorbed by a weighted within-transform, so
    tau is identified only from within-segment, across-boundary contrasts
    (Guide Step 5.2 "boundary-segment fixed effects");
  * cluster-robust (by segment) standard errors.

tau is the jump in the street-network metric when crossing from the higher- to
the lower-grade side.  statsmodels/rdrobust are not in the env, so the WLS +
cluster-robust SEs are done directly with numpy (same convention as 08d).

Outputs (data/tier1/rd/): {slug}_rd.csv ; results/tier1/rd_all.csv (stacked)
Usage:
  python tier1/04_rd_estimate.py --cities chicago --bandwidth 250
  python tier1/04_rd_estimate.py --metrics intersection_density median_block_length_ft
"""
from __future__ import annotations

import argparse
import math
import warnings

import numpy as np
import pandas as pd

from oi_local import METRIC_NAMES
from tier1_common import OI_DIR, RD_DIR, RES, city_slugs

warnings.filterwarnings("ignore")


def rd_one(d: pd.DataFrame, metric: str, h: float):
    d = d.dropna(subset=[metric, "signed_dist", "treat", "seg_id"])
    x = d["signed_dist"].to_numpy(float)
    y = d[metric].to_numpy(float)
    T = d["treat"].to_numpy(float)
    seg = d["seg_id"].to_numpy(str)
    w = np.clip(1.0 - np.abs(x) / h, 0.0, None)        # triangular kernel
    keep = w > 0
    x, y, T, w, seg = x[keep], y[keep], T[keep], w[keep], seg[keep]
    if len(y) < 20 or np.unique(seg).size < 3 or T.min() == T.max():
        return None
    X = np.column_stack([T, x, T * x])                 # a & seg-FE absorbed below

    # weighted within-segment transform (removes intercept + segment FE)
    Xd = np.empty_like(X); yd = np.empty_like(y)
    ok_seg = []
    for g in np.unique(seg):
        gi = seg == g; wi = w[gi]; sw = wi.sum()
        if sw <= 0 or gi.sum() < 2:
            Xd[gi] = 0.0; yd[gi] = 0.0; continue
        Xd[gi] = X[gi] - (wi[:, None] * X[gi]).sum(0) / sw
        yd[gi] = y[gi] - (wi * y[gi]).sum() / sw
        ok_seg.append(g)
    if len(ok_seg) < 3:
        return None

    XtW = Xd.T * w
    A = XtW @ Xd
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    beta = Ainv @ (XtW @ yd)
    resid = yd - Xd @ beta

    # cluster-robust (by segment) covariance
    meat = np.zeros((3, 3))
    for g in np.unique(seg):
        gi = seg == g
        ug = (Xd[gi].T * w[gi]) @ resid[gi]
        meat += np.outer(ug, ug)
    nseg = np.unique(seg).size
    dof = nseg / max(nseg - 1, 1)
    V = dof * (Ainv @ meat @ Ainv)
    tau, se = float(beta[0]), float(math.sqrt(max(V[0, 0], 0.0)))
    t = tau / se if se > 0 else np.nan
    from scipy import stats
    p = float(2 * stats.t.sf(abs(t), df=max(nseg - 1, 1))) if se > 0 else np.nan
    return dict(tau=tau, se=se, t=t, p=p, n=int(len(y)), n_seg=int(nseg),
                y_mean=float(np.average(y, weights=w)), h=h)


def process_city(slug, metrics, bandwidth):
    p = OI_DIR / f"{slug}_oi.parquet"
    if not p.exists():
        print(f"[{slug}] no Oi table; run 03 first"); return None
    df = pd.read_parquet(p)
    print(f"[{slug}] {len(df):,} Oi samples")
    rows = []
    for pair, dp in df.groupby("pair"):
        for decade, dd in dp.groupby("decade"):
            for metric in metrics:
                r = rd_one(dd, metric, bandwidth)
                if r is None:
                    continue
                rows.append(dict(slug=slug, pair=pair, decade=decade,
                                 metric=metric, **r))
    if not rows:
        print("  no RD estimates"); return None
    out = pd.DataFrame(rows)
    out.to_csv(RD_DIR / f"{slug}_rd.csv", index=False)
    # headline: the present/2020 C-D density & block-length jumps
    head = out[(out.metric.isin(["intersection_density", "median_block_length_ft"]))
               & (out.pair == "C-D")]
    for _, r in head.iterrows():
        star = "***" if r.p < .01 else "**" if r.p < .05 else "*" if r.p < .1 else ""
        print(f"  C-D {r.decade:>7} {r.metric:24s} tau={r.tau:+9.3f} "
              f"se={r.se:7.3f} p={r.p:.3f}{star}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=city_slugs())
    ap.add_argument("--metrics", nargs="+", default=METRIC_NAMES)
    ap.add_argument("--bandwidth", type=float, default=300.0,
                    help="RD bandwidth in metres (triangular kernel)")
    args = ap.parse_args()
    allc = []
    for slug in args.cities:
        o = process_city(slug, args.metrics, args.bandwidth)
        if o is not None:
            allc.append(o)
    if allc:
        a = pd.concat(allc, ignore_index=True)
        a.to_csv(RES / "rd_all.csv", index=False)
        print(f"\nsaved -> {RES/'rd_all.csv'}  ({len(a)} rows)")


if __name__ == "__main__":
    main()
