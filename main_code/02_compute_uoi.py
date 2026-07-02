"""Stage 2: Urban Optionality Index (UOI) metrics per tract graph.

Dimensions (higher = better): connectivity = link-node ratio; efficiency =
1 / length-weighted circuity; accessibility = mean 800 m network reach;
equity = 1 - Gini of per-node reach. Also emits morphology features for
Stage 3. Resumable: GEOIDs already in uoi_metrics.parquet are skipped.

Usage: python 02_compute_uoi.py [--limit 20]
"""
from __future__ import annotations

import argparse
import math
import time
import warnings

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from uoi_common import DATA, GRAPH_DIR, OUT_DIR, gini

REACH_CUTOFF_M = 800     # walking distance for opportunity reach
MAX_SOURCES = 500        # cap on Dijkstra sources per tract
MIN_NODES = 5            # tracts below this are flagged, not scored

OUT_PARQUET = OUT_DIR / "uoi_metrics.parquet"


def tract_metrics(geoid: str, rng: np.random.Generator) -> dict:
    G = ox.load_graphml(GRAPH_DIR / f"{geoid}.graphml")
    row: dict = {"GEOID": geoid, "n_nodes": len(G), "n_edges_dir": G.number_of_edges()}
    if len(G) < MIN_NODES:
        row["status"] = "too_small"
        return row

    # --- orientation entropy (needs unprojected graph) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            Gb = ox.bearing.add_edge_bearings(G)
            row["orientation_entropy"] = ox.bearing.orientation_entropy(
                ox.convert.to_undirected(Gb))
        except Exception:
            row["orientation_entropy"] = np.nan

    # --- project, undirect ---
    Gp = ox.project_graph(G)
    Gu = ox.convert.to_undirected(Gp)
    n, m = len(Gu), Gu.number_of_edges()
    xs = nx.get_node_attributes(Gu, "x")
    ys = nx.get_node_attributes(Gu, "y")

    # connectivity: link-node ratio
    row["link_node_ratio"] = m / n

    # degree-based morphology
    degs = dict(Gu.degree())
    row["dead_end_frac"] = sum(1 for d in degs.values() if d == 1) / n
    row["n_intersections"] = sum(1 for d in degs.values() if d >= 3)

    # efficiency: length-weighted average fractional circuity
    tot_len, tot_straight = 0.0, 0.0
    for u, v, data in Gu.edges(data=True):
        if u == v:
            continue
        straight = math.hypot(xs[u] - xs[v], ys[u] - ys[v])
        if straight < 1e-6:
            continue
        tot_len += data.get("length", straight)
        tot_straight += straight
    circuity = (tot_len / tot_straight) if tot_straight > 0 else np.nan
    row["circuity_avg"] = circuity

    # accessibility + equity: 800 m network reach per node
    nodes = list(Gu.nodes)
    sources = nodes if n <= MAX_SOURCES else list(
        rng.choice(nodes, size=MAX_SOURCES, replace=False))
    reach = np.array([
        len(nx.single_source_dijkstra_path_length(
            Gu, s, cutoff=REACH_CUTOFF_M, weight="length")) - 1
        for s in sources
    ], dtype=float)
    row["reach_mean"] = float(reach.mean())
    row["reach_gini"] = gini(reach)
    row["n_reach_sources"] = len(sources)

    # UOI dimensions (higher = better)
    row["uoi_connectivity"] = row["link_node_ratio"]
    row["uoi_efficiency"] = 1.0 / circuity if circuity and not math.isnan(circuity) else np.nan
    row["uoi_accessibility"] = row["reach_mean"]
    row["uoi_equity"] = 1.0 - row["reach_gini"]
    row["status"] = "ok"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max tracts this run")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    done: set[str] = set()
    prior = None
    if OUT_PARQUET.exists():
        prior = pd.read_parquet(OUT_PARQUET)
        done = set(prior["GEOID"])

    geoids = sorted(p.stem for p in GRAPH_DIR.glob("*.graphml") if p.stem not in done)
    if args.limit:
        geoids = geoids[: args.limit]
    print(f"{len(done)} tracts already scored, {len(geoids)} to process")

    rng = np.random.default_rng(args.seed)
    rows, t0 = [], time.time()
    for i, geoid in enumerate(geoids, 1):
        try:
            rows.append(tract_metrics(geoid, rng))
        except Exception as e:  # keep going; record the failure
            rows.append({"GEOID": geoid, "status": f"error: {e}"})
        if i % 25 == 0 or i == len(geoids):
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(geoids)} ({rate:.1f} tracts/s)")

    if rows:
        df = pd.DataFrame(rows)
        if prior is not None:
            df = pd.concat([prior, df], ignore_index=True)
        df.to_parquet(OUT_PARQUET, index=False)
        df.to_csv(OUT_DIR / "uoi_metrics.csv", index=False)
        ok = df[df["status"] == "ok"]
        print(f"\nsaved {len(df)} rows ({len(ok)} scored) -> {OUT_PARQUET}")
        if len(ok):
            print(ok[["uoi_connectivity", "uoi_efficiency",
                      "uoi_accessibility", "uoi_equity"]].describe().round(3))


if __name__ == "__main__":
    main()
