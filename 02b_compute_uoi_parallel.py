"""Stage 2 (spec) PARALLEL driver.

Reuses the exact metric code in 02_compute_uoi_spec.py (single source of truth)
but fans the per-tract work out across CPU cores with multiprocessing. The
single-core version runs ~0.9 tract/s; with N workers this scales ~linearly
(metrics 5 & 6 are CPU-bound Dijkstra/ego-graph, so cores help directly).

Resumable: GEOIDs already in data/outputs/uoi_spec_metrics.parquet are skipped.
Per-tract RNG is seeded by a stable hash of the GEOID, so results are
reproducible regardless of worker scheduling.

Usage:
    python 02b_compute_uoi_parallel.py --workers 14
    python 02b_compute_uoi_parallel.py --workers 14 --states 06,48,17
"""
from __future__ import annotations
import argparse, glob, hashlib, importlib.util, time
import multiprocessing as mp
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from uoi_common import GRAPH_DIR, OUT_DIR, DATA

# import the metric implementation from the (digit-prefixed) spec module
_spec = importlib.util.spec_from_file_location(
    "uoi_spec_impl", str(Path(__file__).with_name("02_compute_uoi_spec.py")))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
tract_metrics = _mod.tract_metrics

OUT = OUT_DIR / "uoi_spec_metrics.parquet"

_TG = None  # tract attrs (geometry+ALAND), shared via fork copy-on-write


def _seed(geoid: str) -> int:
    return int.from_bytes(hashlib.md5(geoid.encode()).digest()[:4], "little")


def _work(geoid: str) -> dict:
    rng = np.random.default_rng(_seed(geoid))
    try:
        geom = _TG.loc[geoid, "geometry"]
        aland = float(_TG.loc[geoid, "ALAND"])
        return tract_metrics(geoid, aland, geom, rng)
    except Exception as e:  # noqa: BLE001 — one bad tract must not kill the pool
        return {"GEOID": geoid, "status": f"error: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    ap.add_argument("--states", default=None, help="comma list e.g. 06,48")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    gdfs = [gpd.read_file(p)[["GEOID", "ALAND", "geometry"]]
            for p in sorted(glob.glob(str(DATA / "tracts_*.gpkg")))]
    tg = pd.concat(gdfs, ignore_index=True).drop_duplicates("GEOID").set_index("GEOID")

    prior, done = None, set()
    if OUT.exists():
        prior = pd.read_parquet(OUT)
        done = set(prior["GEOID"])
    geoids = sorted(p.stem for p in GRAPH_DIR.glob("*.graphml") if p.stem not in done)
    if args.states:
        pre = tuple(args.states.split(","))
        geoids = [g for g in geoids if g[:2] in pre]
    if args.limit:
        geoids = geoids[: args.limit]
    print(f"{len(done)} done, {len(geoids)} to score, {args.workers} workers",
          flush=True)
    if not geoids:
        return

    global _TG
    _TG = tg

    rows, t0 = [], time.time()
    with mp.Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_work, geoids, chunksize=8), 1):
            rows.append(row)
            if i % 200 == 0 or i == len(geoids):
                print(f"  {i}/{len(geoids)} ({i/(time.time()-t0):.1f}/s)", flush=True)

    df = pd.DataFrame(rows)
    if prior is not None:
        df = pd.concat([prior, df], ignore_index=True)
    df = df.drop_duplicates("GEOID", keep="last")
    df.to_parquet(OUT, index=False)
    df.to_csv(OUT_DIR / "uoi_spec_metrics.csv", index=False)
    ok = df[df["status"] == "ok"]
    cols = ["link_node_ratio", "connected_node_ratio", "intersection_density",
            "median_block_length_ft", "walking_circuity", "pedshed_reach"]
    print(f"\nsaved {len(df)} ({len(ok)} ok) -> {OUT}", flush=True)
    if len(ok):
        print(ok[cols].describe().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
