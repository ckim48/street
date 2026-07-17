"""Regime 1 + Regime 2 on the USGS-topo-digitized pre/post networks (all 5 cities).

Runs the Detroit-style analysis for every severed neighborhood using the
`RJMCMC_Ready_Networks` topo graphs (pf_topo) -- real pre/post-highway geometry
digitized from USGS Historical Topographic maps, which unblocks Syracuse, New
Orleans, St. Paul and Miami (OHM had no pre/post coverage there).

R1 (pre-highway optimal search): reuses 20_regime1.optimize on the topo PRE graph
  (reversible-jump add/remove, access objective over HOLC-D landmarks, mu=0.8).
R2 (pre->post severance): DEMOLITION = a topo-PRE street whose geometry is NOT
  covered by the topo-POST network within 25 m (razed / paved over, not merely
  re-gridded).  Reported as % of the neighborhood's pre grid.  NB the topo data
  is NEIGHBORHOOD-scoped, so (unlike Detroit's citywide OHM) there is no citywide
  per-HOLC-grade gradient -- instead the 5-city demolition splits by highway FORM
  (trench razes the grid; elevated viaducts leave the surface streets), the same
  axis Regime 3 found.

Outputs (results/pathfinder/): fig_{slug}_topo_regime1.png,
  fig_{slug}_topo_regime2.png, fig_topo_demolition_compare.png,
  topo_regime1_summary.csv, topo_regime2_summary.csv.
Usage: python pathfinder/24_topo_regimes.py [--cities ...] [--iters 3000]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd
from shapely.ops import unary_union
from shapely.prepared import prep

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pf_style as S
import pf_topo as pt
from pf_common import CITIES, RES

warnings.filterwarnings("ignore")

_spec = importlib.util.spec_from_file_location(
    "r1", Path(__file__).with_name("20_regime1_prehighway.py"))
r1 = importlib.util.module_from_spec(_spec)
sys.modules["r1"] = r1
_spec.loader.exec_module(r1)

# highway form (matches 10_regime3 HW_TYPE); colours from pf_style
HW_TYPE = {"detroit": "trench", "st_paul": "trench", "syracuse": "elevated",
           "new_orleans": "elevated", "miami": "interchange"}
TYPE_COLOR = {"trench": S.ORANGE, "elevated": S.BLUE, "interchange": S.YELLOW}


def mark_demolished(pre, post, buf=25.0):
    """A topo-PRE edge is 'demolished' if <30% of its length is covered by the
    topo-POST network (buffered by `buf` m for ~10-20 m digitizing noise).  This
    counts streets that genuinely disappeared, not streets re-gridded nearby."""
    postbuf = unary_union(post.geometry.values).buffer(buf)
    pre = pre.copy()
    pre["L"] = pre.geometry.length
    pre["surv"] = (pre.geometry.intersection(postbuf).length / pre["L"]).clip(0, 1)
    pre["demolished"] = pre["surv"] < 0.30
    return pre


def draw_r2(slug, pre, post, path):
    cfg = CITIES[slug]
    pre_yr, post_yr = pt.TOPO_YEARS[slug]
    lay = pt.topo_layers(slug)
    omega, barrier, holc_d = lay["omega"], lay["barrier"], lay["holc_d"]
    gone = pre[pre["demolished"]]
    dkm = gone["L"].sum() / 1000
    pkm = pre["L"].sum() / 1000
    # focus the view on the network + Ω, and clip the redline shading to it
    minx, miny, maxx, maxy = omega.union(pre.union_all()).bounds
    mgn = 0.06 * max(maxx - minx, maxy - miny) + 100
    holc_view = holc_d.intersection(omega.buffer(600).envelope)
    fig = plt.figure(figsize=(11.5, 6))
    for i, (edges, yr, ttl, demo) in enumerate(
            [(pre, pre_yr, "PRE", True), (post, post_yr, "POST", False)]):
        ax = fig.add_subplot(1, 2, i + 1)
        if not holc_view.is_empty:
            gpd.GeoSeries([holc_view]).plot(ax=ax, facecolor=S.C["holc_d"], alpha=0.11,
                                            edgecolor=S.C["holc_d"], lw=0.5, zorder=1)
        gpd.GeoSeries([barrier]).plot(ax=ax, facecolor=S.C["barrier"], alpha=0.32,
                                      edgecolor="none", zorder=1)
        edges.plot(ax=ax, color=S.C["base"], lw=0.7, zorder=2)
        gpd.GeoSeries([omega]).plot(ax=ax, facecolor="none", edgecolor=S.C["omega"],
                                    lw=1.0, zorder=3)
        if demo and len(gone):
            gone.plot(ax=ax, color=S.C["demolished"], lw=1.8, zorder=4)
        ax.set_xlim(minx - mgn, maxx + mgn); ax.set_ylim(miny - mgn, maxy + mgn)
        S.title(ax, f"{ttl} {yr}", fontsize=11)
        if ttl == "PRE":
            S.legend(ax, [
                ("line", S.C["demolished"], f"demolished by {post_yr}"),
                ("fill", S.C["holc_d"], "redlined (HOLC-D)"),
                ("band", S.C["barrier"], "highway ROW"),
                ("thin", S.C["base"], f"{yr} topo streets")], fontsize=7.5)
        ax.set_aspect("equal"); ax.set_axis_off()
    fig.suptitle(f"{cfg['city']} — {cfg['neighborhood']} · {cfg['highway']} "
                 f"({HW_TYPE[slug]}) · topo {pre_yr}→{post_yr}\n"
                 f"demolished {dkm:.1f} of {pkm:.1f} km "
                 f"({100*dkm/pkm:.1f}% of the pre-highway grid)",
                 fontsize=12, color=S.INK)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run_r2(slug):
    cfg = CITIES[slug]
    pre_yr, post_yr = pt.TOPO_YEARS[slug]
    pre = mark_demolished(pt.topo_edges(slug, "topopre"), pt.topo_edges(slug, "topopost"))
    post = pt.topo_edges(slug, "topopost")
    pkm = pre["L"].sum() / 1000
    dkm = pre[pre["demolished"]]["L"].sum() / 1000
    draw_r2(slug, pre, post, RES / f"fig_{slug}_topo_regime2.png")
    print(f"  R2 {slug} ({HW_TYPE[slug]}): demolished {dkm:.1f}/{pkm:.1f}km "
          f"({100*dkm/pkm:.1f}%)")
    return dict(slug=slug, city=cfg["city"], neighborhood=cfg["neighborhood"],
                highway=cfg["highway"], form=HW_TYPE[slug],
                pre_year=pre_yr, post_year=post_yr,
                pre_km=round(pkm, 1), demolished_km=round(dkm, 2),
                pct_demolished=round(100 * dkm / pkm, 2) if pkm else 0.0)


def draw_compare(rows, path):
    d = sorted(rows, key=lambda r: -r["pct_demolished"])
    cols = [TYPE_COLOR[r["form"]] for r in d]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = range(len(d))
    ax.bar(x, [r["pct_demolished"] for r in d], color=cols, zorder=3)
    for i, r in enumerate(d):
        ax.text(i, r["pct_demolished"] + 0.15, f"{r['pct_demolished']:.1f}%",
                ha="center", fontsize=9, color=S.INK2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r['city']}\n{r['highway']}" for r in d],
                       fontsize=8, color=S.INK2)
    ax.set_ylabel("% of pre-highway street-km demolished (topo pre→post)",
                  color=S.INK2)
    forms = [f for f in ["trench", "interchange", "elevated"]
             if f in {r["form"] for r in d}]
    ax.legend(handles=[S._handle("fill", TYPE_COLOR[f], f) for f in forms],
              title="highway form", title_fontsize=8, fontsize=8,
              frameon=True, edgecolor=S.GRID, labelcolor=S.INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    S.title(ax, "Street demolition tracks highway FORM: trenches razed the grid, "
                "viaducts spared it", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=pt.TOPO_SLUGS)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--landmarks", type=int, default=28)
    ap.add_argument("--mu", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    r1_rows, r2_rows = [], []
    for slug in a.cities:
        print(f"[{slug}] topo R1 + R2 ...", flush=True)
        G, pos, lay = pt.load_topo_pre(slug)
        row, art = r1.optimize(slug, a.iters, a.landmarks, a.mu, a.seed,
                               graph=(G, pos, lay))
        row["pre_year"] = pt.TOPO_YEARS[slug][0]
        art["row"] = row
        r1.draw(art, RES / f"fig_{slug}_topo_regime1.png")
        r1_rows.append(row)
        print(f"  R1 {slug}: +{row['n_added']}/-{row['n_removed']}  "
              f"access +{row['access_gain']*100:.1f}%", flush=True)
        r2_rows.append(run_r2(slug))

    pd.DataFrame(r1_rows).to_csv(RES / "topo_regime1_summary.csv", index=False)
    pd.DataFrame(r2_rows).to_csv(RES / "topo_regime2_summary.csv", index=False)
    draw_compare(r2_rows, RES / "fig_topo_demolition_compare.png")
    print("\nwrote topo_regime1_summary.csv, topo_regime2_summary.csv, "
          "fig_topo_demolition_compare.png + per-city figures")


if __name__ == "__main__":
    main()
