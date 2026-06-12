"""Stage 3: typology stratification, Pareto frontier, and the deep-analysis sample.

1. Cluster tracts on morphology features (orientation entropy, dead-end
   fraction, circuity, intersection density) with KMeans (k=4) and label the
   clusters gridded / cul_de_sac / organic / hybrid by their feature profiles.
2. Flag the UOI Pareto frontier: tracts not dominated on all four UOI
   dimensions simultaneously (the document's "report as a frontier, don't
   collapse the components").
3. Draw a stratified sample of N tracts, allocated proportionally to stratum
   size (>=1 per stratum), for the RJ-MCMC deep-analysis stage.

Usage:
    python 03_stratified_sample.py --n 1000          # full-scale target
    python 03_stratified_sample.py --n 50            # pilot-scale
"""
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from uoi_common import DATA, OUT_DIR

FEATURES = ["orientation_entropy", "dead_end_frac", "circuity_avg", "intersection_density"]
UOI_COLS = ["uoi_connectivity", "uoi_efficiency", "uoi_accessibility", "uoi_equity"]


def load_metrics() -> pd.DataFrame:
    df = pd.read_parquet(OUT_DIR / "uoi_metrics.parquet")
    df = df[df["status"] == "ok"].copy()

    # join land area from the tract boundary files saved by stage 1
    areas = []
    for gpkg in sorted(DATA.glob("tracts_*.gpkg")):
        areas.append(gpd.read_file(gpkg, ignore_geometry=True)[["GEOID", "ALAND"]])
    area = pd.concat(areas, ignore_index=True).drop_duplicates("GEOID")
    df = df.merge(area, on="GEOID", how="left")
    df["intersection_density"] = df["n_intersections"] / (df["ALAND"] / 1e6)
    df.loc[~np.isfinite(df["intersection_density"]), "intersection_density"] = np.nan
    return df.dropna(subset=FEATURES + UOI_COLS).reset_index(drop=True)


def label_clusters(df: pd.DataFrame) -> dict[int, str]:
    """Assign human-readable typology names to KMeans clusters by profile."""
    prof = df.groupby("cluster")[FEATURES].mean()
    labels: dict[int, str] = {}
    remaining = set(prof.index)

    gridded = prof.loc[list(remaining), "orientation_entropy"].idxmin()
    labels[gridded] = "gridded"
    remaining.discard(gridded)

    culdesac = prof.loc[list(remaining), "dead_end_frac"].idxmax()
    labels[culdesac] = "cul_de_sac"
    remaining.discard(culdesac)

    organic = prof.loc[list(remaining), "circuity_avg"].idxmax()
    labels[organic] = "organic"
    remaining.discard(organic)

    labels[remaining.pop()] = "hybrid"
    return labels


def pareto_front(values: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (maximization on every column)."""
    n = len(values)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        dominates_i = np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1)
        if dominates_i.any():
            nd[i] = False
    return nd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="sample size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_metrics()
    print(f"{len(df)} tracts with complete metrics")

    # --- typology strata ---
    X = StandardScaler().fit_transform(df[FEATURES])
    km = KMeans(n_clusters=4, n_init=10, random_state=args.seed)
    df["cluster"] = km.fit_predict(X)
    df["typology"] = df["cluster"].map(label_clusters(df))
    print("\ntypology profiles (cluster means):")
    print(df.groupby("typology")[FEATURES + ["n_nodes"]].mean().round(3))
    print("\ntypology counts:")
    print(df["typology"].value_counts().to_string())

    # --- UOI Pareto frontier ---
    df["pareto_front"] = pareto_front(df[UOI_COLS].to_numpy())
    print(f"\nUOI Pareto frontier: {df['pareto_front'].sum()} of {len(df)} tracts")

    # --- proportional stratified sample ---
    n_total = min(args.n, len(df))
    shares = df["typology"].value_counts(normalize=True)
    alloc = (shares * n_total).round().astype(int).clip(lower=1)
    while alloc.sum() != n_total:  # fix rounding drift
        k = alloc.idxmax() if alloc.sum() > n_total else shares.idxmax()
        alloc[k] += -1 if alloc.sum() > n_total else 1

    rng = np.random.default_rng(args.seed)
    picks = []
    for typ, k in alloc.items():
        pool = df[df["typology"] == typ]
        picks.append(pool.sample(n=min(k, len(pool)), random_state=rng.integers(1 << 31)))
    sample = pd.concat(picks).sort_values("GEOID")

    df.to_csv(OUT_DIR / "typology_assignments.csv", index=False)
    sample["GEOID"].to_csv(OUT_DIR / "sample_tracts.csv", index=False)
    print(f"\nsampled {len(sample)} tracts "
          f"({dict(sample['typology'].value_counts())}) -> {OUT_DIR / 'sample_tracts.csv'}")


if __name__ == "__main__":
    main()
