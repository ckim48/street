"""PathFinder Regime 1: pre-highway PLAIN optimal-network search.

Regime 1 takes the neighborhood's street grid as it stood the year BEFORE the
interstate was built (construction_year - 1), reconstructed from real historical
OpenHistoricalMap streets (pf_prehist), and asks: how far was the actual redlined
grid from a cost-efficient optimum?  Unlike Regime 3 there is no highway and no
dollar budget -- this is a "plain" reversible-jump search that may BOTH add and
remove streets (Regime 3 is add-only), optimizing access within the redlined
(HOLC-D) fabric against a street-length penalty.

Objective (maximized by RJ-MCMC-in-simulated-annealing):
    J(G) = access_gain(G)  -  MU * (net_length(G) - net_length(G0)) / net_length(G0)
  access_gain = fractional drop in mean network distance from LANDMARK nodes
                (sampled across the redlined HOLC-D neighborhood, farthest-point
                spread) to the rest of the grid -- i.e. how much better the
                community can reach itself.  MU keeps the optimum from collapsing
                to the complete graph (the classic length-vs-accessibility trade).
Moves (reversible jump): ADD a plausible new street (nearby non-adjacent node
  pair <= R_ADD, planar/non-crossing, degree <= DEG_MAX) OR REMOVE an existing
  street, but never one whose removal disconnects the grid.  No highway, no side
  constraint -- the pre-highway community was a single connected fabric.

Redline linkage: landmarks are drawn from nodes inside the HOLC-D polygons, so
J measures accessibility of the graded-"D" (redlined) community specifically --
the population the highway would later sever.

Outputs (results/pathfinder/): fig_{slug}_regime1.png, regime1_summary.csv,
  data/pathfinder/regime1/{slug}_pre_opt.gpkg (added + removed streets).
Usage: python pathfinder/20_regime1_prehighway.py [--cities detroit ...]
       [--iters 3000] [--landmarks 28] [--mu 0.5] [--seed 7]
"""
from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.prepared import prep

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pf_common import CITIES, PF, RES
from pf_graph import dist, edge_crosses
from pf_prehist import build_pre_graph

warnings.filterwarnings("ignore")

R_ADD = 220.0            # max length of a proposed new street (m)
DEG_MAX = 5
BIG = 1e7               # unreachable-pair distance (m) -- should not occur
REG1_DIR = PF / "regime1"
REG1_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------- objective ---------
def pick_landmarks(pos, holc_d, k, rng):
    """k landmark nodes spread across the redlined (HOLC-D) fabric by
    farthest-point sampling; fall back to nodes nearest HOLC-D if too few sit
    strictly inside the polygons."""
    pin = prep(holc_d)
    inside = [n for n, p in pos.items() if pin.contains(Point(p))]
    if len(inside) < k:
        cx, cy = holc_d.centroid.x, holc_d.centroid.y
        inside = sorted(pos, key=lambda n: dist(pos[n], (cx, cy)))[:max(k * 3, 30)]
    # farthest-point spread over `inside`
    inside = list(inside)
    start = inside[rng.integers(len(inside))]
    chosen = [start]
    d = {n: dist(pos[n], pos[start]) for n in inside}
    while len(chosen) < min(k, len(inside)):
        nxt = max(inside, key=lambda n: d[n])
        chosen.append(nxt)
        for n in inside:
            d[n] = min(d[n], dist(pos[n], pos[nxt]))
    return chosen


def access_cost(G, landmarks):
    """Mean network distance between LANDMARK PAIRS (how well key points across
    the redlined community reach each other).  Pairwise (not all-nodes) so a
    single street add/remove moves the metric measurably -- mirrors Regime 3's
    cross-barrier anchor metric, here spread over the whole neighborhood."""
    lset = set(landmarks)
    tot = cnt = 0.0
    for a in landmarks:
        dl = nx.single_source_dijkstra_path_length(G, a, weight="length")
        for b in landmarks:
            if b == a:
                continue
            tot += dl.get(b, BIG)
            cnt += 1
    return tot / max(cnt, 1)


def net_length(G):
    return sum(d["length"] for _, _, d in G.edges(data=True))


# ------------------------------------------------------------- moves ---------
def add_candidates(G, pos, omega):
    """Feasible NEW edges: node pairs within R_ADD, not already edges, planar
    (no crossing), degree < DEG_MAX, midpoint inside Omega."""
    ids = list(pos)
    xy = np.array([pos[n] for n in ids])
    tree = cKDTree(xy)
    pin = prep(omega)
    seen, out = set(), []
    for i, u in enumerate(ids):
        if G.degree(u) >= DEG_MAX:
            continue
        for j in tree.query_ball_point(xy[i], R_ADD):
            v = ids[j]
            if v == u or G.has_edge(u, v) or G.degree(v) >= DEG_MAX:
                continue
            key = (u, v) if u < v else (v, u)
            if key in seen:
                continue
            seen.add(key)
            mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            if not pin.contains(Point(mx, my)):
                continue
            if edge_crosses(G, pos, u, v):
                continue
            out.append((key[0], key[1], dist(pos[u], pos[v])))
    return out


def optimize(slug, iters, n_land, mu, seed):
    cfg = CITIES[slug]
    rng = np.random.default_rng(seed)
    G0, pos, lay = build_pre_graph(slug)
    if G0.number_of_nodes() == 0:
        raise SystemExit(f"{slug}: empty pre-highway graph (no OHM coverage)")
    G = G0.copy()
    omega = lay["omega"]
    land = pick_landmarks(pos, lay["holc_d"], n_land, rng)

    L0 = net_length(G0)
    D0 = access_cost(G0, land)

    def objective(D, L):
        return (D0 - D) / D0 - mu * (L - L0) / L0

    D, L = D0, L0
    J = objective(D, L)
    added, removed = {}, []          # (u,v)->length ; list of (u,v,length)
    cands = add_candidates(G, pos, omega)
    # Temperature must match the objective's per-move ΔJ scale (~1e-4 here): the
    # earlier T0=0.06 made exp(ΔJ/T)≈1 for every move -> a random walk that never
    # exploits (acceptance pinned ~0.95, seed-unstable optimum).  Scaled ~400x
    # lower so acceptance decays from ~explore to ~0 (proper anneal / hill-climb).
    T0, T1 = 2.0e-4, 5.0e-6
    best = dict(J=J, added={}, removed=[], D=D, L=L)
    trace = []                       # (iter, T, J, best_J, accepted, n_edges)

    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / max(iters - 1, 1))
        do_add = (rng.random() < 0.6) or (G.number_of_edges() <= G0.number_of_edges())
        move = None
        if do_add and cands:
            u, v, w = cands[rng.integers(len(cands))]
            if G.has_edge(u, v) or G.degree(u) >= DEG_MAX or G.degree(v) >= DEG_MAX:
                continue
            if edge_crosses(G, pos, u, v):
                continue
            G.add_edge(u, v, length=w)
            move = ("add", u, v, w)
        else:
            # remove a real edge, but never disconnect the grid
            elist = list(G.edges(data="length"))
            u, v, w = elist[rng.integers(len(elist))]
            G.remove_edge(u, v)
            if not nx.has_path(G, u, v):        # would split the two endpoints
                G.add_edge(u, v, length=w)
                continue
            move = ("remove", u, v, w)

        Dn = access_cost(G, land)
        Ln = net_length(G)
        Jn = objective(Dn, Ln)
        accepted = (Jn >= J) or (rng.random() < np.exp((Jn - J) / max(T, 1e-9)))
        trace.append((it, T, Jn, best["J"], int(accepted), G.number_of_edges()))
        if accepted:
            J, D, L = Jn, Dn, Ln
            k = (move[1], move[2]) if move[1] < move[2] else (move[2], move[1])
            if move[0] == "add":
                added[k] = move[3]
                if k in [tuple(sorted((a, b))) for a, b, _ in removed]:
                    removed = [r for r in removed if tuple(sorted((r[0], r[1]))) != k]
            else:
                if k in added:
                    del added[k]
                else:
                    removed.append((move[1], move[2], move[3]))
            if J > best["J"]:
                best = dict(J=J, added=dict(added), removed=list(removed), D=D, L=L)
        else:
            # revert
            if move[0] == "add":
                G.remove_edge(move[1], move[2])
            else:
                G.add_edge(move[1], move[2], length=move[3])

    # rebuild best graph
    Gb = G0.copy()
    for r in best["removed"]:
        if Gb.has_edge(r[0], r[1]):
            Gb.remove_edge(r[0], r[1])
    for (u, v), w in best["added"].items():
        Gb.add_edge(u, v, length=w)

    eff0 = 1.0 / D0
    effb = 1.0 / best["D"]
    row = dict(
        slug=slug, city=cfg["city"], neighborhood=cfg["neighborhood"],
        highway=cfg["highway"], pre_year=cfg["build_start"] - 1,
        n_nodes=G0.number_of_nodes(), n_edges0=G0.number_of_edges(),
        n_added=len(best["added"]), n_removed=len(best["removed"]),
        len0_km=round(L0 / 1000, 3), len_opt_km=round(best["L"] / 1000, 3),
        D0_m=round(D0, 1), Dopt_m=round(best["D"], 1),
        access_gain=round((D0 - best["D"]) / D0, 4),
        eff_gain=round((effb - eff0) / eff0, 4),
        n_landmarks=len(land),
    )
    art = dict(slug=slug, G0=G0, Gb=Gb, pos=pos, lay=lay, land=land,
               added=best["added"], removed=best["removed"], row=row, trace=trace)
    return row, art


# ------------------------------------------------------------- figure --------
def draw(art, path):
    cfg = CITIES[art["slug"]]
    G0, pos, lay = art["G0"], art["pos"], art["lay"]
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    gpd.GeoSeries([lay["omega"]]).plot(ax=ax, facecolor="none", edgecolor="#bbb", lw=1)
    gpd.GeoSeries([lay["holc_d"]]).plot(ax=ax, facecolor="#d73027", alpha=0.12,
                                        edgecolor="#d73027", lw=0.6, zorder=1)
    # highway footprint for reference (built AFTER this network)
    gpd.GeoSeries([lay["barrier"]]).plot(ax=ax, facecolor="#777", alpha=0.10,
                                         edgecolor="none", zorder=1)
    for u, v in G0.edges:
        ax.plot(*zip(pos[u], pos[v]), color="#999", lw=0.6, zorder=2)
    for (u, v, w) in art["removed"]:
        ax.plot(*zip(pos[u], pos[v]), color="#b2182b", lw=1.6, ls=":", zorder=4)
    for (u, v) in art["added"]:
        ax.plot(*zip(pos[u], pos[v]), color="#1a9850", lw=2.4, zorder=5)
    lx = [pos[n][0] for n in art["land"]]
    ly = [pos[n][1] for n in art["land"]]
    ax.plot(lx, ly, "o", color="#2166ac", ms=4, zorder=6)
    r = art["row"]
    ax.set_title(
        f"{cfg['city']} — {cfg['neighborhood']} · pre-{r['pre_year']} (before {cfg['highway']})\n"
        f"R1 optimal search: +{r['n_added']} / -{r['n_removed']} streets, "
        f"access +{r['access_gain']*100:.1f}% "
        f"(len {r['len0_km']}→{r['len_opt_km']}km)", fontsize=9)
    ax.set_aspect("equal"); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_edits(art):
    cfg = CITIES[art["slug"]]
    pos = art["pos"]
    rows = [dict(kind="added", u=u, v=v, length_m=round(w, 1),
                 geometry=LineString([pos[u], pos[v]]))
            for (u, v), w in art["added"].items()]
    rows += [dict(kind="removed", u=u, v=v, length_m=round(w, 1),
                  geometry=LineString([pos[u], pos[v]]))
             for (u, v, w) in art["removed"]]
    if rows:
        gpd.GeoDataFrame(rows, crs=cfg["utm"]).to_file(
            REG1_DIR / f"{art['slug']}_pre_opt.gpkg", layer="edits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=["detroit"])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--landmarks", type=int, default=28)
    ap.add_argument("--mu", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rows = []
    for slug in a.cities:
        print(f"[{slug}] Regime-1 optimal search ...", flush=True)
        row, art = optimize(slug, a.iters, a.landmarks, a.mu, a.seed)
        draw(art, RES / f"fig_{slug}_regime1.png")
        save_edits(art)
        rows.append(row)
        print(f"    +{row['n_added']}/-{row['n_removed']} streets  "
              f"access_gain={row['access_gain']*100:.1f}%  "
              f"len {row['len0_km']}->{row['len_opt_km']}km", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "regime1_summary.csv", index=False)
    print("\nwrote", RES / "regime1_summary.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
