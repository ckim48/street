"""PathFinder Regime 2: PRE -> POST highway severance, per-HOLC-grade.

Regime 2 uses the SAME real historical OpenHistoricalMap source as Regime 1 to
extract the street network at two moments -- the year before the interstate
(build_start-1) and a year after it and its urban-renewal clearance settled
(POST_YEAR) -- and measures what the highway did to the redlined community's grid.

The clean signal is DEMOLITION, not raw street counts: citywide the network GREW
1958->1975 (postwar expansion), which swamps a naive pre/post count.  So we
isolate destruction: a 1958 street is "demolished by POST_YEAR" if its OHM
end_date <= POST_YEAR.  This factors out growth and exposes where the grid was
torn out.

Two deliverables:

 (A) Per-HOLC-grade demolition intensity ("파랑/초록/노랑/빨강 지역").  For each
     redlining grade A(green) B(blue) C(yellow) D(red) we clip the dated 1958
     OHM streets to that grade's polygons and report km alive, km demolished by
     POST_YEAR, and % demolished -- a difference-in-differences across grades.
     We also report the pre-highway intersection density per grade (UOI level).

 (B) Study-neighborhood (Ω) severance.  Inside Black-Bottom-scale Ω we report the
     demolished street length, and the change in intra-community access: mean
     network distance between landmark nodes spread across the redlined (HOLC-D)
     fabric, PRE vs POST (each on its own largest component) -- the community-
     severance the highway imposed.

Outputs (results/pathfinder/): fig_{slug}_regime2.png, regime2_grade_uoi.csv,
  regime2_severance.csv.
Usage: python pathfinder/21_regime2_prepost.py [--slug detroit] [--post-year 1975]
"""
from __future__ import annotations

import argparse
import warnings
from statistics import median

import geopandas as gpd
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pf_common import BND_DIR, CITIES, HOLC_GPKG, RES
from pf_graph import _graph_from_lines, _noded_lines

warnings.filterwarnings("ignore")

M2_PER_KM2 = 1e6
GRADES = [("A", "green", "#1a9850"), ("B", "blue", "#4575b4"),
          ("C", "yellow", "#f6c342"), ("D", "red", "#d73027")]


def _yr(s):
    try:
        return int(str(s)[:4])
    except (ValueError, TypeError):
        return None


def _d(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


# --------------------------------------------------------- UOI metrics -------
def graph_metrics(streets, poly, area_km2):
    """Structural UOI level metrics for the streets clipped to `poly`."""
    g = gpd.clip(streets, poly)
    g = g[~g.geometry.is_empty & g.geometry.notna()]
    empty = dict(n_nodes=0, n_edges=0, len_km=0.0, link_node_ratio=np.nan,
                 connected_node_ratio=np.nan, intersection_density=np.nan,
                 median_block_m=np.nan)
    if len(g) == 0:
        return empty
    G = _graph_from_lines(_noded_lines(g.geometry.values))
    if G.number_of_nodes() == 0:
        return empty
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = dict(G.degree())
    n_inter = sum(1 for dd in deg.values() if dd >= 3)
    n_dead = sum(1 for dd in deg.values() if dd == 1)
    elens = [dd["length"] for _, _, dd in G.edges(data=True)]
    return dict(
        n_nodes=n, n_edges=m, len_km=round(sum(elens) / 1000, 3),
        link_node_ratio=round(m / n, 3),
        connected_node_ratio=round(n_inter / (n_inter + n_dead), 3) if (n_inter + n_dead) else np.nan,
        intersection_density=round(n_inter / area_km2, 1) if area_km2 else np.nan,
        median_block_m=round(median(elens), 1) if elens else np.nan,
    )


def demolition(streets_pre, poly, post_year):
    """km alive at pre-year and km demolished by post_year (end_date <= post_year)
    for the 1958 streets clipped to `poly`."""
    g = gpd.clip(streets_pre, poly)
    g = g[~g.geometry.is_empty & g.geometry.notna()].copy()
    if len(g) == 0:
        return 0.0, 0.0, 0
    g["e"] = g["end"].map(_yr)
    g["L"] = g.geometry.length
    alive = g["L"].sum()
    demo = g[g["e"].notna() & (g["e"] <= post_year)]
    return alive / 1000, demo["L"].sum() / 1000, len(demo)


# --------------------------------------------------- access (severance) ------
def build_graph(streets, clip_poly):
    g = gpd.clip(streets, clip_poly)
    g = g[~g.geometry.is_empty & g.geometry.notna()]
    G = _graph_from_lines(_noded_lines(g.geometry.values))
    if G.number_of_nodes() == 0:
        return G, {}
    comp = max(nx.connected_components(G), key=len)
    G = G.subgraph(comp).copy()
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}
    return G, pos


def landmarks_in(poly, pos, k, rng):
    pin = prep(poly)
    inside = [nn for nn, p in pos.items() if pin.contains(Point(p))]
    if len(inside) < k:
        inside = list(pos)
    start = inside[rng.integers(len(inside))]
    chosen = [start]
    d = {nn: _d(pos[nn], pos[start]) for nn in inside}
    while len(chosen) < min(k, len(inside)):
        nxt = max(inside, key=lambda nn: d[nn])
        chosen.append(nxt)
        for nn in inside:
            d[nn] = min(d[nn], _d(pos[nn], pos[nxt]))
    return chosen


def access(G, land):
    """Mean network distance between landmark pairs both in G's giant component
    (finite); also fraction of pairs with no path (severed)."""
    incomp = [l for l in land if l in G]
    tot = cnt = sev = pairs = 0.0
    for a in incomp:
        dl = nx.single_source_dijkstra_path_length(G, a, weight="length")
        for b in incomp:
            if b == a:
                continue
            pairs += 1
            if b in dl:
                tot += dl[b]
                cnt += 1
            else:
                sev += 1
    return (tot / max(cnt, 1)), (sev / max(pairs, 1))


# ------------------------------------------------------------- figure --------
def draw(slug, pre_s, post_s, lay, grade_df, post_year, path):
    cfg = CITIES[slug]
    pre_yr = cfg["build_start"] - 1
    omega, barrier, holc_d = lay["omega"], lay["barrier"], lay["holc_d"]
    fig = plt.figure(figsize=(15, 5.6))

    # demolished 1958 streets inside Ω (end <= post_year)
    demo_pre = gpd.clip(pre_s, omega).copy()
    demo_pre = demo_pre[~demo_pre.geometry.is_empty & demo_pre.geometry.notna()]
    demo_pre["e"] = demo_pre["end"].map(_yr)
    gone = demo_pre[demo_pre["e"].notna() & (demo_pre["e"] <= post_year)]

    for i, (streets, yr, ttl) in enumerate([(pre_s, pre_yr, "PRE"),
                                            (post_s, post_year, "POST")]):
        ax = fig.add_subplot(1, 3, i + 1)
        gpd.GeoSeries([omega]).plot(ax=ax, facecolor="none", edgecolor="#bbb", lw=1)
        gpd.GeoSeries([holc_d]).plot(ax=ax, facecolor="#d73027", alpha=0.09,
                                     edgecolor="#d73027", lw=0.5)
        gpd.GeoSeries([barrier]).plot(ax=ax, facecolor="#333", alpha=0.22, edgecolor="none")
        gpd.clip(streets, omega).plot(ax=ax, color="#555", lw=0.6)
        if ttl == "PRE" and len(gone):
            gone.plot(ax=ax, color="#d73027", lw=1.8)
        ax.set_title(f"{ttl} {yr}" + ("  (red = demolished by "
                     f"{post_year})" if ttl == "PRE" else ""), fontsize=10)
        ax.set_aspect("equal"); ax.set_axis_off()

    # per-grade demolition-% bar
    ax = fig.add_subplot(1, 3, 3)
    labels = [f"{g}\n({c})" for g, c, _ in GRADES]
    vals = [grade_df.loc[grade_df.grade == g, "pct_demolished"].values for g, _, _ in GRADES]
    vals = [v[0] if len(v) else 0 for v in vals]
    ax.bar(np.arange(len(GRADES)), vals, color=[c3 for _, _, c3 in GRADES])
    ax.set_xticks(np.arange(len(GRADES))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(f"% of 1958 street-km demolished by {post_year}")
    ax.set_title("Demolition intensity by HOLC grade", fontsize=10)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=9)

    fig.suptitle(f"{cfg['city']} — {cfg['neighborhood']} · {cfg['highway']} "
                 f"severance (Regime 2, real OHM {pre_yr}→{post_year})", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="detroit")
    ap.add_argument("--post-year", type=int, default=1975)
    ap.add_argument("--landmarks", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    slug, cfg = a.slug, CITIES[a.slug]
    pre_yr = cfg["build_start"] - 1
    rng = np.random.default_rng(a.seed)

    # cached OHM (Regime-1 fetch already populated these); import lazily so a
    # missing cache triggers a live fetch with the retry/backoff logic.
    from pf_prehist import fetch_ohm_year
    holc = gpd.read_file(HOLC_GPKG)
    det = holc[holc["city"].astype(str).str.contains(cfg["holc_match"], case=False, na=False)
               & holc["state"].astype(str).str.upper().eq(cfg["state"])].to_crs(cfg["utm"])
    grade_poly = {g: unary_union(det[det["grade"].astype(str).str.upper().str.strip() == g]
                                 .geometry.values) for g, _, _ in GRADES}
    all_poly = unary_union([p for p in grade_poly.values() if not p.is_empty])

    print(f"[{slug}] loading OHM {pre_yr} & {a.post_year} (cache) ...", flush=True)
    pre_s = fetch_ohm_year(slug, pre_yr, tag=f"city{pre_yr}", clip_geom=all_poly)
    post_s = fetch_ohm_year(slug, a.post_year, tag=f"city{a.post_year}", clip_geom=all_poly)
    print(f"    pre={len(pre_s)} streets  post={len(post_s)} streets", flush=True)

    # (A) per-grade demolition intensity + pre-highway UOI level
    rows = []
    for g, cname, _ in GRADES:
        poly = grade_poly[g]
        if poly.is_empty:
            continue
        area = poly.area / M2_PER_KM2
        alive_km, demo_km, n_demo = demolition(pre_s, poly, a.post_year)
        m = graph_metrics(pre_s, poly, area)
        rows.append(dict(slug=slug, grade=g, color=cname, area_km2=round(area, 2),
                         alive_km=round(alive_km, 1), demolished_km=round(demo_km, 2),
                         pct_demolished=round(100 * demo_km / alive_km, 2) if alive_km else 0,
                         n_demolished=n_demo,
                         pre_intersection_density=m["intersection_density"],
                         pre_link_node_ratio=m["link_node_ratio"],
                         pre_median_block_m=m["median_block_m"]))
    gdf = pd.DataFrame(rows)
    gdf.to_csv(RES / "regime2_grade_uoi.csv", index=False)
    print("\n  per-grade demolition (1958 street-km removed by "
          f"{a.post_year}):")
    for r in rows:
        print(f"    {r['grade']} ({r['color']:6s}): alive {r['alive_km']:7.1f}km  "
              f"demolished {r['demolished_km']:6.2f}km  ({r['pct_demolished']:4.1f}%)  "
              f"pre-int.density {r['pre_intersection_density']}")

    # (B) Ω severance
    omega = gpd.read_file(BND_DIR / f"{slug}.gpkg", layer="omega").geometry.iloc[0]
    barrier = gpd.read_file(BND_DIR / f"{slug}.gpkg", layer="barrier").geometry.iloc[0]
    holc_d = unary_union(gpd.read_file(BND_DIR / f"{slug}.gpkg", layer="holc_d").geometry.values)
    lay = dict(omega=omega, barrier=barrier, holc_d=holc_d)

    alive_o, demo_o, n_o = demolition(pre_s, omega, a.post_year)
    Gpre, ppre = build_graph(pre_s, omega)
    Gpost, ppost = build_graph(post_s, omega)
    land = landmarks_in(holc_d, ppre, a.landmarks, rng)
    acc_pre, sev_pre = access(Gpre, land)
    land_post = [min(ppost, key=lambda n: _d(ppost[n], ppre[l])) for l in land]
    acc_post, sev_post = access(Gpost, land_post)

    sev = pd.DataFrame([dict(
        slug=slug, pre_year=pre_yr, post_year=a.post_year,
        omega_alive_km=round(alive_o, 1), omega_demolished_km=round(demo_o, 2),
        omega_pct_demolished=round(100 * demo_o / alive_o, 2) if alive_o else 0,
        omega_n_demolished=n_o,
        access_pre_m=round(acc_pre, 1), access_post_m=round(acc_post, 1),
        access_change_pct=round(100 * (acc_post - acc_pre) / acc_pre, 2) if acc_pre else 0,
        n_landmarks=len(land),
    )])
    sev.to_csv(RES / "regime2_severance.csv", index=False)
    print(f"\n  Ω (Black-Bottom-scale): demolished {demo_o:.1f}km of {alive_o:.1f}km "
          f"({100*demo_o/alive_o:.1f}% of the 1958 grid)")
    print(f"  intra-community access (landmark mean dist): "
          f"{acc_pre:.0f}m → {acc_post:.0f}m ({sev.access_change_pct.values[0]:+.1f}%)")

    draw(slug, pre_s, post_s, lay, gdf, a.post_year, RES / f"fig_{slug}_regime2.png")
    print("\nwrote regime2_grade_uoi.csv, regime2_severance.csv, "
          f"fig_{slug}_regime2.png")


if __name__ == "__main__":
    main()
