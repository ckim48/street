"""PathFinder Regime 3: budget-constrained, ADD-ONLY street restoration.

Given a severed neighborhood's present-day network (TIGER local streets) and the
highway barrier B that cut it, find the set of NEW streets that best reconnects
the two sides of B under a real dollar budget, adding roads only (existing
streets are never removed -- the sampler may retract its OWN proposals so the
annealing can explore).

Objective (maximized by RJ-MCMC-in-simulated-annealing, spec 3.0/3.3):
    J(G) = reconnect_gain(G)  -  lambda * (spend / budget)
  reconnect_gain = fractional drop in mean network distance between anchor nodes
                   straddling the barrier (the "community-severance" the highway
                   imposed); lambda keeps the design budget-efficient.
Hard constraints: add-only; cumulative cost <= budget; planarity (no crossing an
  existing street); degree <= DEG_MAX; new edge inside Omega; (optional) no edge
  into a restricted-land polygon `excl` (parks/4f/floodway -- wired, empty in v1).
Cost: c(e) = length(e) * UNIT_PER_M  (default $5M / mile local street, spec 3.1;
  the highway-teardown cost is NOT included -- reported as a separate line).

Outputs (results/pathfinder/): fig_{slug}_restore.png, restore_summary.csv,
  and data/pathfinder/restore/{slug}_added.gpkg (the proposed new streets).
Usage: python pathfinder/10_regime3_restore.py [--cities detroit ...]
       [--iters 6000] [--unit-per-mile 5e6] [--seed 7]
"""
from __future__ import annotations

import argparse
import math
import warnings

import geopandas as gpd
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.prepared import prep

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pf_common import CITIES, PF, RES
from pf_graph import build_modern_graph, dist, edge_crosses, label_sides

warnings.filterwarnings("ignore")

R_ADD = 260.0            # max length of a proposed new street (m)
DEG_MAX = 5
ANCHORS_K = 10           # anchor nodes per side, nearest the barrier
LAMBDA_COST = 0.15       # budget-efficiency weight in J
BIG = 1e7                # unreachable-pair distance penalty (m)
MILE_M = 1609.344
RESTORE_DIR = PF / "restore"
RESTORE_DIR.mkdir(exist_ok=True)


def _hwy_centerline_pts(hwy):
    pts = []
    for part in getattr(hwy, "geoms", [hwy]):
        pts.extend(part.coords)
    return np.asarray(pts, float)


def barrier_anchors(pos, side, hwy):
    """One anchor node per side at each of ANCHORS_K points spread ALONG the
    highway, so reconnection is measured over the whole severance line (not one
    clustered spot -- otherwise a single crossing saturates the metric)."""
    from shapely.ops import linemerge
    line = linemerge(hwy) if hwy.geom_type == "MultiLineString" else hwy
    if line.geom_type == "MultiLineString":
        line = max(line.geoms, key=lambda g: g.length)
    idsA = [n for n in pos if side[n] > 0]
    idsB = [n for n in pos if side[n] < 0]
    xyA = np.array([pos[n] for n in idsA])
    xyB = np.array([pos[n] for n in idsB])
    tA, tB = cKDTree(xyA), cKDTree(xyB)
    A, B = [], []
    for i in range(ANCHORS_K):
        s = line.interpolate((i + 0.5) / ANCHORS_K * line.length)
        pa = np.array([s.x, s.y])
        A.append(idsA[tA.query(pa)[1]])
        B.append(idsB[tB.query(pa)[1]])
    return list(dict.fromkeys(A)), list(dict.fromkeys(B))


def crossing_candidates(G, pos, side, omega):
    """Feasible NEW edges that cross the barrier: opposite-side node pairs within
    R_ADD, not already edges, whose midpoint lies inside Omega."""
    ids = list(pos)
    xy = np.array([pos[n] for n in ids])
    tree = cKDTree(xy)
    pin = prep(omega)
    seen, out = set(), []
    for i, u in enumerate(ids):
        for j in tree.query_ball_point(xy[i], R_ADD):
            v = ids[j]
            if v == u or side[u] == side[v] or G.has_edge(u, v):
                continue
            key = (u, v) if u < v else (v, u)
            if key in seen:
                continue
            seen.add(key)
            mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            from shapely.geometry import Point
            if not pin.contains(Point(mx, my)):
                continue
            out.append((key[0], key[1], dist(pos[u], pos[v])))
    return out


def reconnect_dist(G, A, B):
    tot = cnt = 0
    for a in A:
        dl = nx.single_source_dijkstra_path_length(G, a, weight="length")
        for b in B:
            tot += dl.get(b, BIG)
            cnt += 1
    return tot / max(cnt, 1)


def restore_city(slug, iters, unit_per_m, seed, excl=None):
    cfg = CITIES[slug]
    rng = np.random.default_rng(seed)
    G, pos, lay = build_modern_graph(slug)
    side, _, _ = label_sides(pos, lay["highway"])
    omega = lay["omega"]
    budget = cfg["project_cost"] or 150_000_000
    A, B = barrier_anchors(pos, side, lay["highway"])
    cands = crossing_candidates(G, pos, side, omega)
    pexcl = prep(excl) if excl is not None else None

    D0 = reconnect_dist(G, A, B)
    Gc = G.copy()
    added = {}                       # (u,v) -> (length, cost)
    spend = 0.0

    def J():
        gain = (D0 - reconnect_dist(Gc, A, B)) / D0 if D0 > 0 else 0.0
        return gain, gain - LAMBDA_COST * (spend / budget)

    gain_cur, Jcur = J()
    best = (Jcur, gain_cur, dict(added), spend)
    T0, Tend = 0.05, 0.002

    for it in range(iters):
        T = T0 * (Tend / T0) ** (it / max(iters - 1, 1))
        do_add = (rng.random() < 0.6) or not added
        if do_add and cands:
            u, v, L = cands[rng.integers(len(cands))]
            cost = L * unit_per_m
            if (u, v) in added or Gc.has_edge(u, v):
                continue
            if spend + cost > budget:
                continue
            if Gc.degree(u) >= DEG_MAX or Gc.degree(v) >= DEG_MAX:
                continue
            if edge_crosses(Gc, pos, u, v):
                continue
            if pexcl is not None and pexcl.intersects(LineString([pos[u], pos[v]])):
                continue
            Gc.add_edge(u, v, length=L)
            spend += cost
            g2, J2 = J()
            if J2 - Jcur > 0 or rng.random() < math.exp((J2 - Jcur) / T):
                gain_cur, Jcur = g2, J2
                added[(u, v)] = (L, cost)
            else:
                Gc.remove_edge(u, v)
                spend -= cost
        elif added:
            (u, v), (L, cost) = list(added.items())[rng.integers(len(added))]
            Gc.remove_edge(u, v)
            spend -= cost
            g2, J2 = J()
            if J2 - Jcur > 0 or rng.random() < math.exp((J2 - Jcur) / T):
                gain_cur, Jcur = g2, J2
                del added[(u, v)]
            else:
                Gc.add_edge(u, v, length=L)
                spend += cost
        if Jcur > best[0]:
            best = (Jcur, gain_cur, dict(added), spend)

    Jb, gain_b, added_b, spend_b = best
    row = dict(
        slug=slug, city=cfg["city"], neighborhood=cfg["neighborhood"],
        highway=cfg["highway"], budget_usd=budget,
        n_candidates=len(cands), n_added=len(added_b),
        added_len_km=round(sum(l for l, _ in added_b.values()) / 1000, 3),
        spend_usd=round(spend_b), pct_budget=round(100 * spend_b / budget, 2),
        reconnect_gain=round(gain_b, 4),
        D0_m=round(D0, 1), unit_per_mile=round(unit_per_m * MILE_M),
    )
    return row, dict(slug=slug, G=G, pos=pos, lay=lay, side=side,
                     A=A, B=B, added=added_b, row=row)


def draw(art, path):
    cfg = CITIES[art["slug"]]
    G, pos, lay = art["G"], art["pos"], art["lay"]
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    gpd.GeoSeries([lay["omega"]]).plot(ax=ax, facecolor="none", edgecolor="#bbb", lw=1)
    gpd.GeoSeries([lay["barrier"]]).plot(ax=ax, facecolor="#e74c3c", alpha=0.18,
                                         edgecolor="none")
    for u, v in G.edges:
        ax.plot(*zip(pos[u], pos[v]), color="#888", lw=0.5, zorder=2)
    for (u, v) in art["added"]:
        ax.plot(*zip(pos[u], pos[v]), color="#1a9850", lw=2.6, zorder=4)
    for n in art["A"]:
        ax.plot(*pos[n], "o", color="#2166ac", ms=4, zorder=5)
    for n in art["B"]:
        ax.plot(*pos[n], "o", color="#b2182b", ms=4, zorder=5)
    r = art["row"]
    ax.set_title(f"{cfg['city']} — {cfg['neighborhood']} · {cfg['highway']}\n"
                 f"restore {r['n_added']} crossings, {r['added_len_km']}km, "
                 f"{r['spend_usd']/1e6:.1f}M of {r['budget_usd']/1e6:.0f}M USD "
                 f"({r['pct_budget']}%) · reconnect +{r['reconnect_gain']*100:.1f}%",
                 fontsize=9)
    ax.set_aspect("equal"); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_added(art):
    cfg = CITIES[art["slug"]]
    pos = art["pos"]
    rows = [dict(u=u, v=v, length_m=round(l, 1), cost_usd=round(c),
                 geometry=LineString([pos[u], pos[v]]))
            for (u, v), (l, c) in art["added"].items()]
    if rows:
        gpd.GeoDataFrame(rows, crs=cfg["utm"]).to_file(
            RESTORE_DIR / f"{art['slug']}_added.gpkg", layer="added_streets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=list(CITIES))
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--unit-per-mile", type=float, default=5e6, dest="unit_mile")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    unit_per_m = args.unit_mile / MILE_M

    rows = []
    for slug in args.cities:
        print(f"[{slug}] optimizing ...", flush=True)
        row, art = restore_city(slug, args.iters, unit_per_m, args.seed)
        rows.append(row)
        draw(art, RES / f"fig_{slug}_restore.png")
        save_added(art)
        print(f"    +{row['n_added']} crossings  {row['added_len_km']}km  "
              f"${row['spend_usd']/1e6:.1f}M ({row['pct_budget']}%)  "
              f"reconnect +{row['reconnect_gain']*100:.1f}%  (cands={row['n_candidates']})",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "restore_summary.csv", index=False)
    print("\n" + df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
