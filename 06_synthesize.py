"""Stage 5b: benchmark-calibrated SYNTHESIS of optimal *virtual* street networks
(design-doc Part 3c-i, "high-UOI modes of unconstrained networks — entirely new").

Unlike 05_mcmc_spec.py (counterfactuals seeded from, and bounded by, a real
tract), this builds networks FROM SCRATCH in a blank square domain — tied to no
city — and optimizes them toward the UOI levels the top-1000 actually achieve.

Difference from 05: there is no real reference network, so the energy is an
ABSOLUTE "reach-or-better" goodness toward a target metric vector (the top-1000
medians), with the circuity band [1.2,1.7] from the design doc:
    higher-better dims : reward rising up to the target, then plateau
    block length       : reward falling down to the target, then plateau
    circuity           : reward staying inside the band
    E = sum_i w_i * r_i,  r_i in (-1, 0],  w ~ Dirichlet(1^6)
Maximum E = 0 = "every metric meets the top-1000 benchmark".

Three seed archetypes give diverse modes: gridded / organic (Delaunay) / hybrid.
The RJ-MCMC move + tempering machinery is reused from 04_sampler.py; the metric
evaluator is reused from 05_mcmc_spec.py.

Output (results/synth/): the best virtual network per (seed x chain) as a figure
grid + a CSV of their 6 metrics vs the top-1000 target.

Usage: python 06_synthesize.py [--iters 5000 --side 700 --grid 13 --chains 3]
"""
from __future__ import annotations
import argparse, importlib.util, json, math
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay

from uoi_common import OUT_DIR, ROOT

ROOTD = Path(__file__).resolve().parent
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, ROOTD / path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
s04 = _load("sampler04", "04_sampler.py")
s05 = _load("mcmc_spec05", "05_mcmc_spec.py")
propose, Poly, dist = s04.propose, s04.Poly, s04.dist
SpecEvaluator = s05.SpecEvaluator
FT_PER_M, M2_PER_MILE2 = s05.FT_PER_M, s05.M2_PER_MILE2
CIRC_LO, CIRC_HI, TAU = s05.CIRC_LO, s05.CIRC_HI, s05.TAU

RES = ROOT / "results" / "synth"; RES.mkdir(parents=True, exist_ok=True)
METRICS = s05.METRIC_NAMES
# top-1000 achieved medians = the synthesis target (benchmark)
TARGET = np.array([1.731, 0.964, 2430.0, 32.77, 1.231, 0.018])


def goodness(uoi: np.ndarray, target: np.ndarray) -> np.ndarray:
    """reach-or-better per-dim reward in (-1, 0]; 0 = target met or beaten."""
    v = np.clip(uoi, 1e-9, None); t = np.clip(target, 1e-9, None)
    r = np.empty(6)
    for i in (0, 1, 2, 5):                         # higher-better
        r[i] = math.tanh(min(0.0, math.log(v[i] / t[i])) / TAU)
    r[3] = math.tanh(min(0.0, math.log(t[3] / v[3])) / TAU)   # block: lower-better
    c = v[4]                                       # circuity: inside band
    pen = (math.log(CIRC_LO / c) if c < CIRC_LO else
           math.log(c / CIRC_HI) if c > CIRC_HI else 0.0)
    r[4] = -math.tanh(pen / TAU)
    return r


# --------------------------------------------------------------- seeds
def seed_grid(side, k, rng, jitter=0.12):
    G = nx.Graph(); pos = {}
    step = side / (k - 1)
    for i in range(k):
        for j in range(k):
            u = i * k + j
            x = i * step + rng.normal(0, jitter * step)
            y = j * step + rng.normal(0, jitter * step)
            pos[u] = (float(np.clip(x, 1, side - 1)), float(np.clip(y, 1, side - 1)))
            G.add_node(u)
    for i in range(k):
        for j in range(k):
            u = i * k + j
            if i + 1 < k:
                G.add_edge(u, (i + 1) * k + j)
            if j + 1 < k:
                G.add_edge(u, i * k + j + 1)
    return G, pos


def seed_organic(side, n, rng):
    pts = rng.uniform(side * 0.04, side * 0.96, size=(n, 2))
    tri = Delaunay(pts)
    G = nx.Graph(); pos = {i: (float(pts[i, 0]), float(pts[i, 1])) for i in range(n)}
    G.add_nodes_from(range(n))
    for s in tri.simplices:
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            if dist(pos[a], pos[b]) <= 0.28 * side:    # drop long hull edges
                G.add_edge(int(a), int(b))
    return G, pos


def seed_hybrid(side, k, rng):
    G, pos = seed_grid(side, k, rng)               # left half = grid
    nmax = max(pos) + 1
    npts = (k * k) // 2
    pts = rng.uniform([side * 0.52, side * 0.04], [side * 0.96, side * 0.96],
                      size=(npts, 2))
    tri = Delaunay(pts)
    idmap = {i: nmax + i for i in range(npts)}
    for i in range(npts):
        pos[idmap[i]] = (float(pts[i, 0]), float(pts[i, 1])); G.add_node(idmap[i])
    for s in tri.simplices:
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            if dist(pos[idmap[a]], pos[idmap[b]]) <= 0.28 * side:
                G.add_edge(idmap[int(a)], idmap[int(b)])
    # stitch the two halves: connect nearest grid/organic node pair near the seam
    gnodes = [u for u in G if u < nmax and pos[u][0] > side * 0.40]
    onodes = list(idmap.values())
    for u in gnodes:
        v = min(onodes, key=lambda w: dist(pos[u], pos[w]))
        if 20 <= dist(pos[u], pos[v]) <= 0.2 * side and not G.has_edge(u, v):
            G.add_edge(u, v)
    return G, pos


def clean_seed(G, pos, poly_sh):
    """keep nodes inside domain, largest component; set straight-line lengths."""
    from shapely.geometry import Point
    keep = [u for u in G if poly_sh.contains(Point(pos[u]))]
    G = G.subgraph(keep).copy()
    if G.number_of_nodes() == 0:
        return G, pos
    comp = max(nx.connected_components(G), key=len)
    G = G.subgraph(comp).copy()
    for a, b in G.edges:
        G.edges[a, b]["length"] = dist(pos[a], pos[b])
    pos = {u: pos[u] for u in G.nodes}
    return G, pos


# --------------------------------------------------------------- one chain
def synth_chain(seed_kind, side, k, iters, n_temps, anchors_k, sharp, rng):
    from shapely.geometry import box
    poly_sh = box(0, 0, side, side)
    if seed_kind == "grid":
        G0, pos0 = seed_grid(side, k, rng)
    elif seed_kind == "organic":
        G0, pos0 = seed_organic(side, k * k, rng)
    else:
        G0, pos0 = seed_hybrid(side, k, rng)
    G0, pos0 = clean_seed(G0, pos0, poly_sh)
    poly = Poly(poly_sh)
    n0, m0 = len(G0), G0.number_of_edges()
    if n0 < 8:
        return None
    area_mi2 = (side * side) / M2_PER_MILE2
    a_ids = rng.choice(list(G0.nodes), size=min(anchors_k, n0), replace=False)
    ev = SpecEvaluator(np.array([pos0[u] for u in a_ids]), area_mi2)
    w = rng.dirichlet(np.ones(6))

    def energy(uoi):
        return float(np.dot(w, goodness(uoi, TARGET)))

    betas = np.geomspace(1.0, 0.18, n_temps)
    u0 = ev(G0, pos0)
    states = [{"G": G0.copy(), "pos": dict(pos0), "E": energy(u0), "uoi": u0}
              for _ in range(n_temps)]
    next_id = max(G0.nodes) + 1
    best = {"E": states[0]["E"], "G": G0.copy(), "pos": dict(pos0), "uoi": u0}
    for it in range(iters):
        for t in range(n_temps):
            s = states[t]
            prop = propose(s["G"], s["pos"], poly, rng, n0, m0, next_id)
            if prop is None:
                continue
            G2, pos2, logH, next_id = prop
            u2 = ev(G2, pos2); E2 = energy(u2)
            if math.log(rng.uniform() + 1e-300) < sharp * betas[t] * (E2 - s["E"]) + logH:
                states[t] = {"G": G2, "pos": pos2, "E": E2, "uoi": u2}
                if t == 0 and E2 > best["E"]:
                    best = {"E": E2, "G": G2.copy(), "pos": dict(pos2), "uoi": u2}
        if it % 20 == 0 and n_temps > 1:
            t = int(rng.integers(n_temps - 1))
            a, b = states[t], states[t + 1]
            if math.log(rng.uniform() + 1e-300) < sharp * (betas[t] - betas[t + 1]) * (b["E"] - a["E"]):
                states[t], states[t + 1] = b, a
    best["seed"] = seed_kind
    return best


def fig_networks(results, side):
    n = len(results); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.6),
                             facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, results):
        G, pos = r["G"], r["pos"]
        ax.add_collection(LineCollection([[pos[u], pos[v]] for u, v in G.edges],
                                         colors="#1f5fbf", linewidths=0.8))
        u = r["uoi"]
        ax.set_title(f"{r['seed']}  E={r['E']:.2f}\n"
                     f"LNR {u[0]:.2f} CNR {u[1]:.2f} dens {u[2]:.0f}\n"
                     f"blk {u[3]:.0f}ft circ {u[4]:.2f} ped {u[5]:.3f}",
                     fontsize=7.5)
        ax.set_xlim(-10, side + 10); ax.set_ylim(-10, side + 10)
        ax.set_aspect("equal"); ax.set_axis_off()
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.suptitle("Synthetic optimal virtual networks (city-free) — calibrated to "
                 "top-1000 UOI benchmark", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RES / "fig_synth_networks.png", dpi=150); plt.close(fig)
    print(f"saved {RES/'fig_synth_networks.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--side", type=float, default=700.0, help="domain side (m)")
    ap.add_argument("--grid", type=int, default=13, help="grid nodes per side")
    ap.add_argument("--temps", type=int, default=4)
    ap.add_argument("--chains", type=int, default=3, help="chains per seed kind")
    ap.add_argument("--anchors", type=int, default=12)
    ap.add_argument("--sharp", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    results = []
    for kind in ("grid", "organic", "hybrid"):
        for c in range(args.chains):
            r = synth_chain(kind, args.side, args.grid, args.iters, args.temps,
                            args.anchors, args.sharp, rng)
            if r is not None:
                results.append(r)
                print(f"{kind} chain{c}: E={r['E']:.3f} uoi="
                      f"[{', '.join(f'{x:.3g}' for x in r['uoi'])}]", flush=True)
    # keep the best chain per seed kind for the figure
    best_per = {}
    for r in results:
        if r["seed"] not in best_per or r["E"] > best_per[r["seed"]]["E"]:
            best_per[r["seed"]] = r
    fig_set = sorted(results, key=lambda r: -r["E"])[:9]
    fig_networks(fig_set, args.side)

    import csv
    with open(RES / "synth_metrics.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seed", "E"] + METRICS)
        wr.writerow(["TARGET(top1000)", ""] + [f"{x:.3g}" for x in TARGET])
        for r in sorted(results, key=lambda r: -r["E"]):
            wr.writerow([r["seed"], f"{r['E']:.4f}"] + [f"{x:.4g}" for x in r["uoi"]])
    print(f"saved {RES/'synth_metrics.csv'}  ({len(results)} networks)")


if __name__ == "__main__":
    main()
