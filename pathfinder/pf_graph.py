"""PathFinder shared graph utilities.

Build the noded modern street graph for a city inside Omega (TIGER local
streets, split at intersections), label nodes by which side of the highway
barrier they fall on, and the small planar-geometry helpers the Regime-3
restoration sampler reuses (mirrors 04_sampler.py so the two stay consistent).
"""
from __future__ import annotations

import math

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.ops import unary_union

from pf_common import BND_DIR


# ------------------------------------------------------- planar geometry ----
def dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _ccw(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def segments_cross(p1, p2, p3, p4) -> bool:
    """Proper intersection of open segments (shared endpoints excluded by caller)."""
    d1 = _ccw(*p3, *p4, *p1)
    d2 = _ccw(*p3, *p4, *p2)
    d3 = _ccw(*p1, *p2, *p3)
    d4 = _ccw(*p1, *p2, *p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def edge_crosses(G, pos, u, v, skip=()) -> bool:
    """Does segment u-v properly cross any existing edge (excluding incident ones)?"""
    pu, pv = pos[u], pos[v]
    for a, b in G.edges:
        if a in (u, v) or b in (u, v) or (a, b) in skip or (b, a) in skip:
            continue
        if segments_cross(pu, pv, pos[a], pos[b]):
            return True
    return False


# ------------------------------------------------- graph from noded lines ---
def _graph_from_lines(geoms, snap_tol: float = 3.0) -> nx.Graph:
    """Undirected graph from projected LineStrings; endpoints within snap_tol
    metres merge to one node.  Feed geometries already NODED at intersections
    (see build_modern_graph) so endpoints are the true junctions."""
    G = nx.Graph()
    pts, refs = [], []
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        for part in getattr(geom, "geoms", [geom]):
            coords = list(part.coords)
            if len(coords) < 2:
                continue
            pts.append(coords[0]); refs.append(part)
            pts.append(coords[-1]); refs.append(part)
    if not pts:
        return G
    arr = np.asarray(pts, float)
    tree = cKDTree(arr)
    node_of = -np.ones(len(arr), dtype=int)
    nid = 0
    for i in range(len(arr)):
        if node_of[i] >= 0:
            continue
        grp = tree.query_ball_point(arr[i], snap_tol)
        for j in grp:
            if node_of[j] < 0:
                node_of[j] = nid
        cx, cy = arr[grp].mean(axis=0)
        G.add_node(nid, x=float(cx), y=float(cy))
        nid += 1
    for k in range(0, len(refs), 2):
        part = refs[k]
        u, v = int(node_of[k]), int(node_of[k + 1])
        if u == v:
            continue
        L = float(part.length)
        if G.has_edge(u, v):
            if L < G.edges[u, v]["length"]:
                G.edges[u, v]["length"] = L
        else:
            G.add_edge(u, v, length=L)
    return G


def _noded_lines(geoms):
    """unary_union splits the road soup at every intersection -> clean nodes."""
    merged = unary_union([g for g in geoms if g is not None and not g.is_empty])
    parts = getattr(merged, "geoms", [merged])
    return [p for p in parts if p.geom_type == "LineString" and p.length > 0]


def build_modern_graph(slug):
    """(G, pos, layers) for a city: G = largest connected component of the local
    (non-highway) TIGER streets inside Omega, noded at intersections.
    layers = dict(omega, barrier, highway) shapely geometries (metric CRS)."""
    gpkg = BND_DIR / f"{slug}.gpkg"
    omega = gpd.read_file(gpkg, layer="omega").geometry.iloc[0]
    barrier = gpd.read_file(gpkg, layer="barrier").geometry.iloc[0]
    hwy = unary_union(gpd.read_file(gpkg, layer="highway").geometry.values)
    modern = gpd.read_file(gpkg, layer="modern_roads")
    local = modern[~modern["is_hwy"]]

    G = _graph_from_lines(_noded_lines(local.geometry.values))
    if G.number_of_nodes() == 0:
        return G, {}, dict(omega=omega, barrier=barrier, highway=hwy)
    comp = max(nx.connected_components(G), key=len)
    G = G.subgraph(comp).copy()
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}
    return G, pos, dict(omega=omega, barrier=barrier, highway=hwy)


def label_sides(pos, hwy):
    """Split nodes into the two sides of the highway by its principal axis.
    Returns (side dict node->+1/-1, centroid c, unit normal n)."""
    coords = []
    for part in getattr(hwy, "geoms", [hwy]):
        coords.extend(part.coords)
    P = np.asarray(coords, float)
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    d = vt[0]                                   # highway main direction
    n = np.array([-d[1], d[0]])                 # perpendicular
    side = {k: (1 if (np.asarray(p) - c) @ n >= 0 else -1) for k, p in pos.items()}
    return side, c, n
