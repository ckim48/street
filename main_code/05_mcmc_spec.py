"""Stage 5: Reversible-Jump MCMC (parallel-tempered) optimal-network search
that targets the design-doc *6-metric* UOI Index (the "UOI Index" table), run
over the top-1000 UOI tracts.

This is the spec counterpart of 04_sampler.py. The RJ-MCMC machinery (planar
moves, Hastings ratios, feasibility, tempering) is REUSED verbatim from
04_sampler.py; only the evaluator and the energy change so that the search is
driven by the six design-doc metrics + their recommended bounds/directions:

  1 link_node_ratio        m/n                                    Higher (rec >=1.4)
  2 connected_node_ratio   deg>=3 / (deg>=3 + deg==1)             Higher (rec >=0.7)
  3 intersection_density   (deg>=3) / mile^2  (tract ALAND)       Higher (rec >140)
  4 median_block_length_ft median straight edge length, feet      Lower  (rec <=600)
  5 walking_circuity       mean net/straight over anchor pairs    Band   (1.2-1.7)
  6 pedshed_reach          reachable street length <=400 m /      Higher
                           disk area, averaged over anchors

FAST surrogate (the only way to afford millions of evaluations):
  - metrics 1-4 are exact O(n)/O(m) on the candidate simple graph;
  - metrics 5-6 share ONE multi-source Dijkstra from a fixed set of `anchors`
    (coords snapped to the nearest current node), so circuity uses anchor-pair
    network/straight distance and pedshed uses the per-anchor reachable street
    length within 400 m. (Stage-2's full h3 lattice / 500-OD version is the
    ground truth; this is its cheap MCMC-time proxy, re-scored identically for
    the real network so the comparison stays apples-to-apples.)

Energy:  E(G) = sum_i w_i * tanh( x_i / TAU ),  w ~ Dirichlet(1^6) per chain.
  x_i is a direction-aware, scale-free log-improvement vs. the real network:
  higher-better dims -> log(v/v_real); block -> log(v_real/v); circuity ->
  reduction of the band-violation log-penalty. tanh saturates each dim so no
  single metric runs away — the optimum is "lift every metric toward its
  recommended bound", i.e. exactly the image's criteria.

Outputs (per tract, under data/outputs/sampler_spec/):
  {geoid}_w{w}_r{rep}.pkl   chain payload (best counterfactual G+pos, real G,
                            posterior improvement samples, traces, diagnostics)
  summary.json              distance-to-frontier, R-hat, accept/swap, best E

Usage:
  python 05_mcmc_spec.py --geoids 36061007300 36061006700 \
      --iters 4000 --temps 4 --weights 2 --replicas 2 --procs 6
  python 05_mcmc_spec.py --top 1000 --iters 6000        # drive the whole top-N
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from uoi_common import DATA, GRAPH_DIR, OUT_DIR

# ---- reuse the RJ-MCMC move/geometry machinery from 04_sampler.py ----------
# (module name starts with a digit, so import it by path)
_spec = importlib.util.spec_from_file_location(
    "sampler04", Path(__file__).resolve().parent / "04_sampler.py")
s04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s04)
propose = s04.propose
Poly = s04.Poly
dist = s04.dist

SAMPLER_DIR = OUT_DIR / "sampler_spec"
SAMPLER_DIR.mkdir(exist_ok=True)

# spec constants (kept identical to 02_compute_uoi_spec.py)
M2_PER_MILE2 = 2_589_988.110336
FT_PER_M = 3.280839895
REACH_M = 400.0
DISK_AREA = math.pi * REACH_M ** 2
CIRC_LO, CIRC_HI = 1.2, 1.7     # recommended walking-circuity band
CIRC_STRAIGHT_MIN = 100.0       # ignore anchor pairs closer than this (m)
TAU = 0.5                       # per-dim tanh saturation scale (log units)

METRIC_NAMES = ["link_node_ratio", "connected_node_ratio", "intersection_density",
                "median_block_length_ft", "walking_circuity", "pedshed_reach"]


# ---------------------------------------------------------------- evaluator
class SpecEvaluator:
    """Fast surrogate for the six design-doc metrics on a candidate graph.

    `area_mile2` (tract ALAND) is fixed; `anchors` are fixed coordinates snapped
    to the nearest current node at each call (so circuity/pedshed are stable
    under node shifts)."""

    def __init__(self, anchor_coords: np.ndarray, area_mile2: float):
        self.anchors = anchor_coords          # (k, 2)
        self.area_mile2 = max(area_mile2, 1e-9)

    def __call__(self, G: nx.Graph, pos: dict) -> np.ndarray:
        nodes = list(G.nodes)
        n = len(nodes)
        index = {u: i for i, u in enumerate(nodes)}
        xy = np.array([pos[u] for u in nodes])
        edges = list(G.edges)
        m = len(edges)
        eu = np.array([index[a] for a, b in edges])
        ev = np.array([index[b] for a, b in edges])
        lengths = np.array([G.edges[a, b]["length"] for a, b in edges])
        degs = np.bincount(np.concatenate([eu, ev]), minlength=n)
        n_inter = int((degs >= 3).sum())
        n_dead = int((degs == 1).sum())

        lnr = m / n
        cnr = n_inter / (n_inter + n_dead) if (n_inter + n_dead) else 0.0
        inter_density = n_inter / self.area_mile2
        block_ft = float(np.median(lengths)) * FT_PER_M if m else np.nan

        # one multi-source Dijkstra from the snapped anchors
        adj = csr_matrix(
            (np.concatenate([lengths, lengths]),
             (np.concatenate([eu, ev]), np.concatenate([ev, eu]))),
            shape=(n, n))
        d2 = ((xy[None, :, :] - self.anchors[:, None, :]) ** 2).sum(axis=2)
        aidx = d2.argmin(axis=1)
        dmat = dijkstra(adj, directed=False, indices=aidx)          # (k, n)

        # 5 walking circuity over anchor pairs
        net = dmat[:, aidx]
        euc = np.sqrt(((xy[aidx][None, :, :] - xy[aidx][:, None, :]) ** 2).sum(axis=2))
        iu = np.triu_indices(len(aidx), k=1)
        net_p, euc_p = net[iu], euc[iu]
        ok = np.isfinite(net_p) & (euc_p > CIRC_STRAIGHT_MIN) & (net_p >= euc_p)
        circuity = float((net_p[ok] / euc_p[ok]).mean()) if ok.sum() else 1.0

        # 6 pedshed: per-anchor reachable street length within 400 m
        reached = dmat <= REACH_M                                   # (k, n)
        reach_len = (reached[:, eu] & reached[:, ev]) @ lengths     # (k,)
        pedshed = float(reach_len.mean()) / DISK_AREA

        return np.array([lnr, cnr, inter_density, block_ft, circuity, pedshed])


def _circ_penalty(c: float) -> float:
    """Log-penalty for leaving the recommended circuity band [1.2, 1.7]."""
    if c < CIRC_LO:
        return math.log(CIRC_LO / max(c, 1e-9))
    if c > CIRC_HI:
        return math.log(c / CIRC_HI)
    return 0.0


def improvement_vector(uoi: np.ndarray, u_real: np.ndarray) -> np.ndarray:
    """Direction-aware, scale-free, tanh-saturated per-dim improvement vs real.
    All six components are 'higher is better' and 0 at the real network."""
    v = np.clip(uoi, 1e-9, None)
    r = np.clip(u_real, 1e-9, None)
    x = np.empty(6)
    x[0] = math.log(v[0] / r[0])              # link-node ratio  (higher)
    x[1] = math.log(v[1] / r[1])              # connected-node   (higher)
    x[2] = math.log(v[2] / r[2])              # intersection density (higher)
    x[3] = math.log(r[3] / v[3])              # block length     (lower)
    x[4] = _circ_penalty(r[4]) - _circ_penalty(v[4])   # circuity band
    x[5] = math.log(v[5] / r[5])              # pedshed reach    (higher)
    return np.tanh(x / TAU)


# --------------------------------------------------------------- load state
def _area_mile2_for(geoid: str) -> float:
    for gpkg in sorted(DATA.glob("tracts_*.gpkg")):
        t = gpd.read_file(gpkg, columns=["GEOID", "ALAND"])
        hit = t[t.GEOID == geoid]
        if len(hit):
            return float(hit.ALAND.iloc[0]) / M2_PER_MILE2
    return np.nan


def run_chain(job):
    (geoid, w, w_idx, replica, iters, n_temps, anchors_k, sharp, seed) = job
    rng = np.random.default_rng(seed)
    G0, pos0, poly_sh = s04.load_state(geoid)
    poly = Poly(poly_sh)
    area_mile2 = _area_mile2_for(geoid)
    n0, m0 = len(G0), G0.number_of_edges()

    a_ids = rng.choice(list(G0.nodes), size=min(anchors_k, len(G0)), replace=False)
    ev = SpecEvaluator(np.array([pos0[u] for u in a_ids]), area_mile2)
    u_real = ev(G0, pos0)

    def energy(uoi):
        return float(np.dot(w, improvement_vector(uoi, u_real)))

    betas = np.geomspace(1.0, 0.18, n_temps)
    e0 = energy(u_real)
    states = [{"G": G0.copy(), "pos": dict(pos0), "E": e0, "uoi": u_real.copy()}
              for _ in range(n_temps)]
    next_id = max(G0.nodes) + 1

    accept = np.zeros(n_temps); tries = np.zeros(n_temps)
    swap_acc = swap_try = 0
    trace, samples = [], []
    best = {"E": e0, "G": G0.copy(), "pos": dict(pos0), "uoi": u_real.copy()}
    burn = iters // 2

    for it in range(iters):
        for t in range(n_temps):
            s = states[t]
            prop = propose(s["G"], s["pos"], poly, rng, n0, m0, next_id)
            tries[t] += 1
            if prop is None:
                continue
            G2, pos2, logH, next_id = prop
            uoi2 = ev(G2, pos2)
            E2 = energy(uoi2)
            if math.log(rng.uniform() + 1e-300) < sharp * betas[t] * (E2 - s["E"]) + logH:
                states[t] = {"G": G2, "pos": pos2, "E": E2, "uoi": uoi2}
                accept[t] += 1
                if t == 0 and E2 > best["E"]:
                    best = {"E": E2, "G": G2.copy(), "pos": dict(pos2), "uoi": uoi2}
        if it % 20 == 0 and n_temps > 1:
            t = int(rng.integers(n_temps - 1))
            a, b = states[t], states[t + 1]
            swap_try += 1
            if math.log(rng.uniform() + 1e-300) < sharp * (betas[t] - betas[t + 1]) * (b["E"] - a["E"]):
                states[t], states[t + 1] = b, a
                swap_acc += 1
        if it % 10 == 0:
            trace.append(states[0]["E"])
        if it >= burn and it % 25 == 0:
            samples.append(improvement_vector(states[0]["uoi"], u_real))

    out = {
        "geoid": geoid, "w_idx": w_idx, "replica": replica, "weights": w.tolist(),
        "u_real": u_real.tolist(), "metric_names": METRIC_NAMES, "trace": trace,
        "samples": np.array(samples).tolist(),
        "accept_rate": (accept / np.maximum(tries, 1)).tolist(),
        "swap_rate": swap_acc / max(swap_try, 1),
        "best_E": best["E"], "best_uoi": best["uoi"].tolist(),
    }
    with open(SAMPLER_DIR / f"{geoid}_w{w_idx}_r{replica}.pkl", "wb") as f:
        pickle.dump({**out, "best_G": best["G"], "best_pos": best["pos"],
                     "G_real": G0, "pos_real": pos0}, f)
    return out


# ------------------------------------------------------- frontier / R-hat
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


def pareto_mask(v: np.ndarray) -> np.ndarray:
    nd = np.ones(len(v), dtype=bool)
    for i in range(len(v)):
        if nd[i] and (np.all(v >= v[i], axis=1) & np.any(v > v[i], axis=1)).any():
            nd[i] = False
    return nd


def hypervolume_shortfall(front, real, rng, n_mc=120_000):
    """Relative hypervolume shortfall of the real network vs the achievable
    frontier, in the 6-D improvement space.

    Every dimension is a tanh-saturated, direction-aware improvement over the
    real network, so improvements live in (-1, 1) and the real network is the
    origin. The reference point (box floor) is the FIXED nadir -1 in every dim
    (the tanh lower bound), not a cloud-dependent value — otherwise, for the
    elite top tracts (already best-in-class on link-node/connected-node/
    density/block, so the posterior never beats them there), the real point
    would sit on the box floor and the shortfall would degenerate to 1.0.

    `front` must be the Pareto front of {posterior cloud} U {real}; the real
    network is included so a fully Pareto-optimal real gives dtf = 0 and a real
    that the search can expand beyond gives dtf in (0, 1)."""
    ref = np.full(real.shape[0], -1.0)
    hi = np.maximum(front.max(axis=0), real) + 1e-9
    pts = rng.uniform(ref, hi, size=(n_mc, real.shape[0]))
    dom_front = np.zeros(n_mc, dtype=bool)
    for f in front:
        dom_front |= np.all(pts <= f, axis=1)
    dom_real = np.all(pts <= real, axis=1)
    hv_f, hv_r = dom_front.mean(), dom_real.mean()
    return float(1.0 - hv_r / hv_f) if hv_f > 0 else np.nan


def resolve_geoids(args) -> list[str]:
    if args.geoids:
        return args.geoids
    top = pd.read_parquet(Path("results/top1000/top1000_uoi.parquet"))
    return top.GEOID.astype(str).head(args.top).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geoids", nargs="+")
    ap.add_argument("--top", type=int, default=1000,
                    help="if --geoids absent, drive the first N top-1000 tracts")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--temps", type=int, default=4)
    ap.add_argument("--weights", type=int, default=2)
    ap.add_argument("--replicas", type=int, default=2)
    ap.add_argument("--anchors", type=int, default=12)
    ap.add_argument("--sharp", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--outdir", type=str, default=None,
                    help="override output dir (default data/outputs/sampler_spec); "
                         "use a separate dir for pilot runs so the main results "
                         "are not overwritten")
    ap.add_argument("--resume", action="store_true",
                    help="skip tracts already in summary.json")
    args = ap.parse_args()

    if args.outdir:
        global SAMPLER_DIR
        SAMPLER_DIR = Path(args.outdir)
        SAMPLER_DIR.mkdir(parents=True, exist_ok=True)
        print(f"output dir -> {SAMPLER_DIR}", flush=True)

    geoids = resolve_geoids(args)
    summary = {}
    sum_path = SAMPLER_DIR / "summary.json"
    if args.resume and sum_path.exists():
        summary = json.loads(sum_path.read_text())
        geoids = [g for g in geoids if g not in summary]
    print(f"{len(geoids)} tract(s) to optimize", flush=True)

    rng = np.random.default_rng(args.seed)
    jobs = []
    for geoid in geoids:
        for w_idx in range(args.weights):
            w = rng.dirichlet(np.ones(6))
            for rep in range(args.replicas):
                jobs.append((geoid, w, w_idx, rep, args.iters, args.temps,
                             args.anchors, args.sharp, int(rng.integers(1 << 31))))
    print(f"{len(jobs)} chains "
          f"({len(geoids)}x{args.weights}w x{args.replicas}r), "
          f"{args.temps} temps x {args.iters} iters", flush=True)

    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        results = list(ex.map(run_chain, jobs))

    for geoid in geoids:
        rs = [r for r in results if r["geoid"] == geoid]
        clouds = [np.array(r["samples"]) for r in rs if r["samples"]]
        if not clouds:
            continue
        cloud = np.vstack(clouds)
        real = np.zeros(6)                     # real network = origin in impr. space
        cand = np.vstack([cloud, real])        # include real among frontier candidates
        front = cand[pareto_mask(cand)]
        dtf = hypervolume_shortfall(front, real, rng)
        rhats = {}
        for w_idx in {r["w_idx"] for r in rs}:
            traces = [r["trace"] for r in rs if r["w_idx"] == w_idx]
            rhats[f"w{w_idx}"] = round(split_rhat(traces), 3)
        best = max(rs, key=lambda r: r["best_E"])
        summary[geoid] = {
            "distance_to_frontier": round(dtf, 4),
            "rhat": rhats,
            "accept_rate_cold": round(float(np.mean([r["accept_rate"][0] for r in rs])), 3),
            "swap_rate": round(float(np.mean([r["swap_rate"] for r in rs])), 3),
            "frontier_size": int(len(front)),
            "posterior_samples": int(len(cloud)),
            "u_real": [round(x, 4) for x in best["u_real"]],
            "best_uoi": [round(x, 4) for x in best["best_uoi"]],
            "best_E": round(best["best_E"], 4),
            "metric_names": METRIC_NAMES,
        }
        print(f"{geoid}: dtf={dtf:.3f} rhat={rhats} "
              f"acc={summary[geoid]['accept_rate_cold']} "
              f"swap={summary[geoid]['swap_rate']} bestE={best['best_E']:.3f}",
              flush=True)
        sum_path.write_text(json.dumps(summary, indent=2))   # checkpoint each tract

    print(f"\nsaved -> {sum_path}  ({len(summary)} tracts)", flush=True)


if __name__ == "__main__":
    main()
