"""Shared Tier 1 helpers: build a routable graph from dated line vectors, and
compute a *local* Oi (UOI) on a graph clip.

The metric definitions are kept identical to 02_compute_uoi_spec.py so city Oi
values are comparable to the national Tier 0 index (Guide Step 3.2 "use the
exact same Oi definition across all cities").  The only difference is scope:
here a metric is evaluated on the subgraph inside a small disk around a boundary
sample point, not over a whole tract.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

M2_PER_MILE2 = 2_589_988.110336
FT_PER_M = 3.280839895
REACH_M = 400.0
DISK_AREA = math.pi * REACH_M ** 2
CIRC_LO, CIRC_HI = 1.2, 1.7

METRIC_NAMES = ["link_node_ratio", "connected_node_ratio", "intersection_density",
                "median_block_length_ft", "walking_circuity", "pedshed_reach"]


# --------------------------------------------------- graph from line vectors
def graph_from_lines(geoms, snap_tol: float = 2.0) -> nx.Graph:
    """Build an undirected topological graph from projected (meter) LineStrings.

    Endpoints within `snap_tol` metres are merged into one node (so the vector
    soup becomes a routable network).  Edge weight = geometry length in metres.
    Node attrs x, y are metric coordinates.
    """
    G = nx.Graph()
    pts, refs = [], []          # endpoint coord -> (geom_index, which_end)
    for gi, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        for part in getattr(geom, "geoms", [geom]):
            coords = list(part.coords)
            if len(coords) < 2:
                continue
            for end in (coords[0], coords[-1]):
                pts.append(end); refs.append((gi, part, coords))
    if not pts:
        return G
    arr = np.asarray(pts, float)
    tree = cKDTree(arr)
    # union-find style snapping to a canonical node id per cluster
    node_of = -np.ones(len(arr), dtype=int)
    nid = 0
    for i in range(len(arr)):
        if node_of[i] >= 0:
            continue
        grp = tree.query_ball_point(arr[i], snap_tol)
        for j in grp:
            if node_of[j] < 0:
                node_of[j] = nid
        node_of[i] = nid
        cx, cy = arr[grp].mean(axis=0)
        G.add_node(nid, x=float(cx), y=float(cy))
        nid += 1
    # add edges (two endpoints per geometry, consecutive in the pts list)
    for k in range(0, len(refs), 2):
        gi, part, coords = refs[k]
        u, v = node_of[k], node_of[k + 1]
        if u == v:
            continue
        L = part.length
        if G.has_edge(u, v):
            if L < G.edges[u, v]["length"]:
                G.edges[u, v]["length"] = L
        else:
            G.add_edge(u, v, length=float(L))
    return G


def clip_graph(G: nx.Graph, tree: cKDTree, node_ids, node_xy,
               center: tuple[float, float], radius: float) -> nx.Graph:
    """Subgraph of G induced by nodes within `radius` of `center`."""
    idx = tree.query_ball_point(center, radius)
    keep = [node_ids[i] for i in idx]
    return G.subgraph(keep)


# ----------------------------------------------------------- local Oi vector
def local_oi(H: nx.Graph, disk_area_m2: float, rng=None) -> dict:
    """Six-metric Oi on a graph clip H (nodes carry metric x,y; edges 'length').
    `disk_area_m2` is the sampling-disk area used for the density metric."""
    n = H.number_of_nodes(); m = H.number_of_edges()
    if n < 5 or m < 4:
        return {k: np.nan for k in METRIC_NAMES} | {"n_nodes": n, "n_edges": m}
    degs = dict(H.degree())
    n_inter = sum(1 for d in degs.values() if d >= 3)
    n_dead = sum(1 for d in degs.values() if d == 1)
    px = nx.get_node_attributes(H, "x"); py = nx.get_node_attributes(H, "y")

    lnr = m / n
    denom = n_inter + n_dead
    cnr = n_inter / denom if denom else np.nan
    inter_density = n_inter / (disk_area_m2 / M2_PER_MILE2)
    elens = [d.get("length", math.hypot(px[u]-px[v], py[u]-py[v]))
             for u, v, d in H.edges(data=True) if u != v]
    block_ft = float(np.median(elens)) * FT_PER_M if elens else np.nan

    # local walking circuity: shortest-path vs straight for a few node pairs
    nodes = list(H.nodes)
    rng = rng or np.random.default_rng(0)
    ratios = []
    srcs = rng.choice(nodes, size=min(len(nodes), 6), replace=False)
    for s in srcs:
        dl = nx.single_source_dijkstra_path_length(H, s, weight="length")
        for t, d in dl.items():
            if t == s:
                continue
            straight = math.hypot(px[s]-px[t], py[s]-py[t])
            if straight > 50:
                ratios.append(d / straight)
    circ = float(np.median(ratios)) if ratios else np.nan

    # pedshed: reachable street length within 400 m of the clip centroid node
    cx = float(np.mean(list(px.values()))); cy = float(np.mean(list(py.values())))
    cn = nodes[int(np.argmin([(px[u]-cx)**2 + (py[u]-cy)**2 for u in nodes]))]
    ego = nx.ego_graph(H, cn, radius=REACH_M, distance="length")
    reach_len = sum(d.get("length", 0.0) for _, _, d in ego.edges(data=True))
    pedshed = reach_len / DISK_AREA

    return {"link_node_ratio": lnr, "connected_node_ratio": cnr,
            "intersection_density": inter_density, "median_block_length_ft": block_ft,
            "walking_circuity": circ, "pedshed_reach": pedshed,
            "n_nodes": n, "n_edges": m}
