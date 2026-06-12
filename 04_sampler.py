"""Stage 4: Reversible-Jump MCMC sampler with parallel tempering.

For each deep-analysis tract, explores the space of physically plausible
street networks inside the tract polygon and finds high-UOI counterfactuals.

State        simple undirected graph; nodes have coordinates, edges are
             straight segments (real curved geometry is abstracted away, and
             the REAL network is re-scored under the same abstraction so the
             comparison is apples-to-apples).
Moves        shift node | add edge | remove edge | subdivide edge (add node)
             | merge degree-2 node (remove node) — each paired with its
             reverse, with counted Hastings ratios (see APPROXIMATIONS).
Constraints  planarity (no segment crossings), connectedness, degree <= 5,
             nodes inside tract polygon, min node spacing, edge length bounds.
Target       pi(G) proportional to exp(SHARP * beta * E(G)) where
             E(G) = sum_i w_i * ln(u_i(G) / u_i(G_real)) over the 4 UOI dims
             (log-ratio vs. the real network -> scale-free). Each chain draws
             its weight vector w ~ Dirichlet(1,1,1,1); replicas share w so
             Gelman-Rubin R-hat is well-defined per weight vector.
Tempering    geometric beta ladder, state swaps between adjacent temperatures.
Outputs      per tract: posterior UOI cloud (cold chains), Pareto frontier,
             real network's relative hypervolume shortfall ("distance to
             frontier"), best counterfactual network, convergence diagnostics.

APPROXIMATIONS (documented for the paper, to tighten later):
- subdivide/merge position-proposal density treated as uniform-disk both ways;
- anchor set for accessibility/efficiency is a fixed set of coordinates
  snapped to the nearest current node at evaluation time;
- accessibility = mean reachable street length within 800 m walk from anchors;
  efficiency = 1 / mean anchor-pair circuity (network / euclidean distance).

Usage:
    python 04_sampler.py --geoids 06075012802 06075021600 06075030500 \
        --iters 12000 --temps 4 --weights 2 --replicas 2
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from uoi_common import DATA, GRAPH_DIR, OUT_DIR, gini

SAMPLER_DIR = OUT_DIR / "sampler"
SAMPLER_DIR.mkdir(exist_ok=True)

# geometric constraints (meters)
MIN_SPACING = 15.0
MIN_LEN = 20.0
MAX_LEN = 250.0
DEG_MAX = 5
R_CAND = 180.0     # max length of a proposed new edge
JITTER_R = 20.0    # subdivide-position jitter radius
SHIFT_SIGMA = 12.0
REACH_CUTOFF = 800.0
SIZE_GUARD = 3.0   # node/edge count may not exceed 3x the initial network

MOVES = ["shift", "add_edge", "remove_edge", "add_node", "remove_node"]
MOVE_P = [0.40, 0.15, 0.15, 0.15, 0.15]


# ----------------------------------------------------------------- geometry
def _ccw(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def segments_cross(p1, p2, p3, p4) -> bool:
    """Proper intersection of open segments (shared endpoints excluded by caller)."""
    d1 = _ccw(*p3, *p4, *p1)
    d2 = _ccw(*p3, *p4, *p2)
    d3 = _ccw(*p1, *p2, *p3)
    d4 = _ccw(*p1, *p2, *p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def edge_crosses(G, pos, u, v, skip=()):
    """Does segment u-v properly cross any existing edge (excluding incident ones)?"""
    pu, pv = pos[u], pos[v]
    for a, b in G.edges:
        if a in (u, v) or b in (u, v) or (a, b) in skip or (b, a) in skip:
            continue
        if segments_cross(pu, pv, pos[a], pos[b]):
            return True
    return False


def dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


# ---------------------------------------------------------------- evaluator
class UOIEvaluator:
    """Fast 4-dim UOI for a candidate graph via one scipy multi-source Dijkstra."""

    def __init__(self, anchor_coords: np.ndarray):
        self.anchors = anchor_coords  # (k, 2) fixed coordinates

    def __call__(self, G: nx.Graph, pos: dict) -> np.ndarray:
        nodes = list(G.nodes)
        index = {u: i for i, u in enumerate(nodes)}
        xy = np.array([pos[u] for u in nodes])
        eu = np.array([index[a] for a, b in G.edges])
        ev = np.array([index[b] for a, b in G.edges])
        lengths = np.array([G.edges[a, b]["length"] for a, b in G.edges])

        n = len(nodes)
        adj = csr_matrix(
            (np.concatenate([lengths, lengths]),
             (np.concatenate([eu, ev]), np.concatenate([ev, eu]))),
            shape=(n, n),
        )
        # snap fixed anchor coordinates to nearest current nodes
        d2 = ((xy[None, :, :] - self.anchors[:, None, :]) ** 2).sum(axis=2)
        aidx = d2.argmin(axis=1)
        dmat = dijkstra(adj, directed=False, indices=aidx)

        # accessibility: mean reachable street length within REACH_CUTOFF
        reached = dmat <= REACH_CUTOFF  # (k, n)
        reach_len = (reached[:, eu] & reached[:, ev]) @ lengths  # (k,)

        # efficiency: 1 / mean circuity over anchor pairs
        net = dmat[:, aidx]
        euc = np.sqrt(((xy[aidx][None, :, :] - xy[aidx][:, None, :]) ** 2).sum(axis=2))
        iu = np.triu_indices(len(aidx), k=1)
        net_p, euc_p = net[iu], euc[iu]
        ok = np.isfinite(net_p) & (euc_p > 100.0) & (net_p >= euc_p)
        circuity = float((net_p[ok] / euc_p[ok]).mean()) if ok.sum() else 1.0

        return np.array([
            G.number_of_edges() / n,        # connectivity
            1.0 / max(circuity, 1.0),       # efficiency
            float(reach_len.mean()),        # accessibility
            1.0 - gini(reach_len),          # equity
        ])


# ------------------------------------------------------------------- moves
def feasible_new_node(G, pos, p, poly, ignore=()):
    if not poly.contains_point(p):
        return False
    for u in G.nodes:
        if u in ignore:
            continue
        if dist(p, pos[u]) < MIN_SPACING:
            return False
    return True


class Poly:
    """Pickle-friendly point-in-polygon wrapper around shapely."""

    def __init__(self, shapely_poly):
        self._poly = shapely_poly

    def contains_point(self, p) -> bool:
        from shapely.geometry import Point
        return self._poly.contains(Point(p))


def edge_candidates(G, pos, u):
    """Nodes v that could be connected to u by a new feasible edge."""
    out = []
    for v in G.nodes:
        if v == u or G.has_edge(u, v):
            continue
        if G.degree[u] >= DEG_MAX or G.degree[v] >= DEG_MAX:
            continue
        L = dist(pos[u], pos[v])
        if not (MIN_LEN <= L <= min(MAX_LEN, R_CAND)):
            continue
        if not edge_crosses(G, pos, u, v):
            out.append(v)
    return out


def removable_edges(G):
    """Edges whose removal keeps the graph connected and degrees >= 1."""
    bridges = set(nx.bridges(G))
    out = []
    for a, b in G.edges:
        if (a, b) in bridges or (b, a) in bridges:
            continue
        if G.degree[a] <= 1 or G.degree[b] <= 1:
            continue
        out.append((a, b))
    return out


def deg2_mergeable(G, pos):
    """Degree-2 nodes that can be removed by joining their two neighbors."""
    out = []
    for u in G.nodes:
        if G.degree[u] != 2:
            continue
        v, w = list(G.neighbors(u))
        if G.has_edge(v, w):
            continue
        L = dist(pos[v], pos[w])
        if not (MIN_LEN <= L <= MAX_LEN):
            continue
        if not edge_crosses(G, pos, v, w, skip=((u, v), (u, w))):
            out.append(u)
    return out


def propose(G, pos, poly, rng, n0, m0, next_id):
    """Return (G2, pos2, log_hastings, next_id) or None if proposal infeasible."""
    move = rng.choice(len(MOVES), p=MOVE_P)
    name = MOVES[move]

    if name == "shift":
        u = list(G.nodes)[rng.integers(len(G))]
        p_new = (pos[u][0] + rng.normal(0, SHIFT_SIGMA),
                 pos[u][1] + rng.normal(0, SHIFT_SIGMA))
        if not feasible_new_node(G, pos, p_new, poly, ignore=(u,)):
            return None
        for v in G.neighbors(u):
            L = dist(p_new, pos[v])
            if not (MIN_LEN <= L <= MAX_LEN):
                return None
        pos2 = dict(pos)
        pos2[u] = p_new
        for v in G.neighbors(u):
            if edge_crosses(G, pos2, u, v, skip=[(u, w) for w in G.neighbors(u)]):
                return None
        G2 = G.copy()
        for v in G2.neighbors(u):
            G2.edges[u, v]["length"] = dist(p_new, pos2[v])
        return G2, pos2, 0.0, next_id

    if name == "add_edge":
        if G.number_of_edges() >= m0 * SIZE_GUARD:
            return None
        u = list(G.nodes)[rng.integers(len(G))]
        cand = edge_candidates(G, pos, u)
        if not cand:
            return None
        v = cand[rng.integers(len(cand))]
        G2 = G.copy()
        G2.add_edge(u, v, length=dist(pos[u], pos[v]))
        # forward: pick u then v among candidates (symmetrized over both ends)
        cu, cv = len(cand), len(edge_candidates(G, pos, v))
        q_fwd = (1 / (len(G) * cu)) + (1 / (len(G) * max(cv, 1)))
        q_rev = 1 / max(len(removable_edges(G2)), 1)
        return G2, pos, math.log(q_rev / q_fwd), next_id

    if name == "remove_edge":
        rem = removable_edges(G)
        if not rem:
            return None
        a, b = rem[rng.integers(len(rem))]
        G2 = G.copy()
        G2.remove_edge(a, b)
        ca, cb = len(edge_candidates(G2, pos, a)) + 1, len(edge_candidates(G2, pos, b)) + 1
        q_fwd = 1 / len(rem)
        q_rev = (1 / (len(G2) * ca)) + (1 / (len(G2) * cb))
        return G2, pos, math.log(q_rev / q_fwd), next_id

    if name == "add_node":
        if len(G) >= n0 * SIZE_GUARD:
            return None
        edges = list(G.edges)
        a, b = edges[rng.integers(len(edges))]
        mid = ((pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2)
        r, th = JITTER_R * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
        p_new = (mid[0] + r * math.cos(th), mid[1] + r * math.sin(th))
        if not feasible_new_node(G, pos, p_new, poly):
            return None
        La, Lb = dist(p_new, pos[a]), dist(p_new, pos[b])
        if not (MIN_LEN <= La <= MAX_LEN and MIN_LEN <= Lb <= MAX_LEN):
            return None
        u = next_id
        pos2 = dict(pos)
        pos2[u] = p_new
        G2 = G.copy()
        G2.remove_edge(a, b)
        G2.add_node(u)
        G2.add_edge(a, u, length=La)
        G2.add_edge(u, b, length=Lb)
        if edge_crosses(G2, pos2, a, u, skip=((u, b),)) or \
           edge_crosses(G2, pos2, u, b, skip=((a, u),)):
            return None
        q_fwd = (1 / len(edges)) * (1 / (math.pi * JITTER_R ** 2))
        q_rev = 1 / max(len(deg2_mergeable(G2, pos2)), 1)
        return G2, pos2, math.log(q_rev / q_fwd), next_id + 1

    # remove_node: merge a degree-2 node
    merge = deg2_mergeable(G, pos)
    if not merge:
        return None
    u = merge[rng.integers(len(merge))]
    v, w = list(G.neighbors(u))
    G2 = G.copy()
    G2.remove_node(u)
    G2.add_edge(v, w, length=dist(pos[v], pos[w]))
    pos2 = {k: q for k, q in pos.items() if k != u}
    q_fwd = 1 / len(merge)
    q_rev = (1 / G2.number_of_edges()) * (1 / (math.pi * JITTER_R ** 2))
    return G2, pos2, math.log(q_rev / q_fwd), next_id


# ------------------------------------------------------------------- chain
def load_state(geoid: str):
    """Real tract network -> simple straight-edge graph + polygon (projected)."""
    G0 = ox.load_graphml(GRAPH_DIR / f"{geoid}.graphml")
    Gp = ox.project_graph(G0)
    crs = Gp.graph["crs"]
    Gu = nx.Graph()
    for u, v in ox.convert.to_undirected(Gp).edges():
        if u == v:
            continue
        Gu.add_edge(u, v)
    pos = {u: (Gp.nodes[u]["x"], Gp.nodes[u]["y"]) for u in Gu.nodes}
    # largest component, straight-line lengths, drop too-short edges' duplicates
    comp = max(nx.connected_components(Gu), key=len)
    Gu = Gu.subgraph(comp).copy()
    for a, b in Gu.edges:
        Gu.edges[a, b]["length"] = dist(pos[a], pos[b])
    pos = {u: pos[u] for u in Gu.nodes}

    tracts = None
    for gpkg in sorted(DATA.glob("tracts_*.gpkg")):
        t = gpd.read_file(gpkg)
        if (t.GEOID == geoid).any():
            tracts = t
            break
    poly = tracts[tracts.GEOID == geoid].to_crs(crs).geometry.iloc[0].buffer(20)
    return Gu, pos, poly


def run_chain(job):
    """One tempered chain (all temperatures stepped each iteration)."""
    (geoid, w, w_idx, replica, iters, n_temps, anchors_k, sharp, seed) = job
    rng = np.random.default_rng(seed)
    G0, pos0, poly_sh = load_state(geoid)
    poly = Poly(poly_sh)
    n0, m0 = len(G0), G0.number_of_edges()

    a_ids = rng.choice(list(G0.nodes), size=min(anchors_k, len(G0)), replace=False)
    ev = UOIEvaluator(np.array([pos0[u] for u in a_ids]))
    u_real = ev(G0, pos0)
    u_real = np.clip(u_real, 1e-9, None)

    def energy(uoi):
        return float(np.dot(w, np.log(np.clip(uoi, 1e-9, None) / u_real)))

    betas = np.geomspace(1.0, 0.18, n_temps)
    states = [{"G": G0.copy(), "pos": dict(pos0), "E": energy(u_real),
               "uoi": u_real.copy()} for _ in range(n_temps)]
    next_id = max(G0.nodes) + 1

    accept = np.zeros(n_temps)
    tries = np.zeros(n_temps)
    swap_acc, swap_try = 0, 0
    trace, samples = [], []
    best = {"E": -np.inf, "G": None, "pos": None, "uoi": None}
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
        if it % 20 == 0 and n_temps > 1:  # swap attempt
            t = int(rng.integers(n_temps - 1))
            a, b = states[t], states[t + 1]
            swap_try += 1
            if math.log(rng.uniform() + 1e-300) < sharp * (betas[t] - betas[t + 1]) * (b["E"] - a["E"]):
                states[t], states[t + 1] = b, a
                swap_acc += 1
        if it % 10 == 0:
            trace.append(states[0]["E"])
        if it >= burn and it % 25 == 0:
            samples.append(states[0]["uoi"])

    out = {
        "geoid": geoid, "w_idx": w_idx, "replica": replica, "weights": w.tolist(),
        "u_real": u_real.tolist(), "trace": trace,
        "samples": np.array(samples).tolist(),
        "accept_rate": (accept / np.maximum(tries, 1)).tolist(),
        "swap_rate": swap_acc / max(swap_try, 1),
        "best_E": best["E"], "best_uoi": None if best["G"] is None else best["uoi"].tolist(),
    }
    pkl = SAMPLER_DIR / f"{geoid}_w{w_idx}_r{replica}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({**out, "best_G": best["G"], "best_pos": best["pos"],
                     "G_real": G0, "pos_real": pos0}, f)
    return out


# ------------------------------------------------------- frontier / R-hat
def split_rhat(chains: list[list[float]]) -> float:
    """Split Gelman-Rubin R-hat over >=2 same-target traces (2nd half)."""
    seqs = []
    for c in chains:
        h = np.asarray(c[len(c) // 2:], dtype=float)
        seqs += [h[: len(h) // 2], h[len(h) // 2:]]
    L = min(len(s) for s in seqs)
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


def hypervolume_shortfall(front: np.ndarray, real: np.ndarray, rng, n_mc=200_000):
    """Relative hypervolume shortfall of `real` vs the frontier (MC estimate)."""
    allpts = np.vstack([front, real])
    # reference point at the origin: every UOI dimension has a natural zero
    lo = np.zeros(allpts.shape[1])
    hi = allpts.max(axis=0) + 1e-12
    pts = rng.uniform(lo, hi, size=(n_mc, allpts.shape[1]))
    dom_front = np.zeros(n_mc, dtype=bool)
    for f in front:
        dom_front |= np.all(pts <= f, axis=1)
    dom_real = np.all(pts <= real, axis=1)
    hv_f, hv_r = dom_front.mean(), dom_real.mean()
    return float(1.0 - hv_r / hv_f) if hv_f > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geoids", nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--temps", type=int, default=4)
    ap.add_argument("--weights", type=int, default=2, help="weight vectors per tract")
    ap.add_argument("--replicas", type=int, default=2, help="chains per weight vector")
    ap.add_argument("--anchors", type=int, default=12)
    ap.add_argument("--sharp", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    jobs = []
    for geoid in args.geoids:
        for w_idx in range(args.weights):
            w = rng.dirichlet(np.ones(4))
            for rep in range(args.replicas):
                jobs.append((geoid, w, w_idx, rep, args.iters, args.temps,
                             args.anchors, args.sharp,
                             int(rng.integers(1 << 31))))
    print(f"{len(jobs)} chains ({len(args.geoids)} tracts x {args.weights} weights "
          f"x {args.replicas} replicas), {args.temps} temps x {args.iters} iters each")

    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        results = list(ex.map(run_chain, jobs))

    summary = {}
    for geoid in args.geoids:
        rs = [r for r in results if r["geoid"] == geoid]
        cloud = np.vstack([np.array(r["samples"]) for r in rs if r["samples"]])
        u_real = np.array(rs[0]["u_real"])
        front = cloud[pareto_mask(cloud)]
        dtf = hypervolume_shortfall(front, u_real, rng)
        rhats = {}
        for w_idx in {r["w_idx"] for r in rs}:
            traces = [r["trace"] for r in rs if r["w_idx"] == w_idx]
            rhats[f"w{w_idx}"] = round(split_rhat(traces), 3)
        summary[geoid] = {
            "distance_to_frontier": round(dtf, 4),
            "rhat": rhats,
            "accept_rate_cold": round(float(np.mean([r["accept_rate"][0] for r in rs])), 3),
            "swap_rate": round(float(np.mean([r["swap_rate"] for r in rs])), 3),
            "frontier_size": int(len(front)),
            "posterior_samples": int(len(cloud)),
            "u_real": [round(x, 4) for x in u_real],
            "best_E": round(max(r["best_E"] for r in rs), 4),
        }
        print(f"\n{geoid}: dtf={dtf:.3f} rhat={rhats} "
              f"acc={summary[geoid]['accept_rate_cold']} swap={summary[geoid]['swap_rate']}")

    with open(SAMPLER_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {SAMPLER_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
