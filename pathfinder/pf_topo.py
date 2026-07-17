"""Loader for the user-supplied USGS-topo-digitized pre/post networks.

The `RJMCMC_Ready_Networks` bundle (data/pathfinder/topo/) provides, for the five
original severed neighborhoods, machine-readable street graphs at four epochs
digitized from USGS Historical Topographic Maps:
  topopre_<city>            pre-highway map  (e.g. Detroit 1954, Rondo 1951)
  topopost_<city>           post-highway map (first sheet showing the interstate)
  prehighway_fallback_<city> TIGER 2025 minus the interstate (surviving grid)
  modern_tiger2025_<city>   present-day TIGER
This unblocks Regime 1/2 for the four cities OHM could not support (Syracuse,
New Orleans, St. Paul, Miami) with real pre/post historical geometry.

Bundle city names map to our slugs: treme→new_orleans, rondo→st_paul,
overtown→miami (detroit, syracuse unchanged).  Study-area Ω / barrier B / HOLC-D
come from the existing boundaries gpkg (00_boundaries); the *graph* is the topo
network.
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
from shapely.ops import unary_union

from pf_common import BND_DIR, CITIES, PF

TOPO = PF / "topo"
SLUG2BUNDLE = {"detroit": "detroit", "syracuse": "syracuse",
               "new_orleans": "treme", "st_paul": "rondo", "miami": "overtown"}
TOPO_SLUGS = list(SLUG2BUNDLE)
# USGS PRE / POST map edition years actually digitized (README §1)
TOPO_YEARS = {"detroit": (1954, 1968), "syracuse": (1958, 1973),
              "new_orleans": (1966, 1973), "st_paul": (1951, 1969),
              "miami": (1950, 1970)}


def _dir(slug, epoch):
    b = SLUG2BUNDLE[slug]
    return TOPO / f"{epoch}_{b}", f"{epoch}_{b}"


def topo_edges(slug, epoch):
    """GeoDataFrame (city UTM) of the epoch's edges; columns incl. length_m, src."""
    d, name = _dir(slug, epoch)
    g = gpd.read_file(d / f"{name}_edges.geojson")
    return g.to_crs(CITIES[slug]["utm"])


def load_topo_graph(slug, epoch="topopre"):
    """(G, pos): largest connected component of the topo network in city UTM.
    G edges carry 'length' (m); pos maps node->(x_utm, y_utm)."""
    d, name = _dir(slug, epoch)
    H = nx.read_graphml(d / f"{name}.graphml")
    G = nx.Graph()
    for n, a in H.nodes(data=True):
        G.add_node(int(n), x=float(a["x_utm"]), y=float(a["y_utm"]))
    for u, v, a in H.edges(data=True):
        u, v = int(u), int(v)
        if u == v:
            continue
        L = float(a.get("length_m", 0.0))
        if G.has_edge(u, v):
            if L < G.edges[u, v]["length"]:
                G.edges[u, v]["length"] = L
        else:
            G.add_edge(u, v, length=L)
    if G.number_of_nodes() == 0:
        return G, {}
    comp = max(nx.connected_components(G), key=len)
    G = G.subgraph(comp).copy()
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}
    return G, pos


def bundle_omega(slug):
    """The community-documented boundary the topo networks were clipped to
    (bundle boundaries.geojson), in city UTM -- matches the network extent."""
    b = SLUG2BUNDLE[slug]
    bnd = gpd.read_file(TOPO / "boundaries.geojson")
    sel = bnd[bnd["city"].astype(str) == b]
    if len(sel) == 0:
        return None
    return sel.to_crs(CITIES[slug]["utm"]).geometry.iloc[0]


def topo_layers(slug):
    """Ω (bundle boundary, matches the network) + barrier / highway / holc_d
    (from the boundaries gpkg / Mapping Inequality)."""
    gpkg = BND_DIR / f"{slug}.gpkg"
    omega = bundle_omega(slug)
    if omega is None:
        omega = gpd.read_file(gpkg, layer="omega").geometry.iloc[0]
    barrier = gpd.read_file(gpkg, layer="barrier").geometry.iloc[0]
    hwy = unary_union(gpd.read_file(gpkg, layer="highway").geometry.values)
    holc_d = unary_union(gpd.read_file(gpkg, layer="holc_d").geometry.values)
    return dict(omega=omega, barrier=barrier, highway=hwy, holc_d=holc_d)


def load_topo_pre(slug):
    """(G, pos, layers) mirroring pf_prehist.build_pre_graph, from the topo PRE map."""
    G, pos = load_topo_graph(slug, "topopre")
    return G, pos, topo_layers(slug)
