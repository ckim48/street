"""PathFinder Step 0: severed-neighborhood boundary Omega + highway barrier B.

For each city:
  * highway   = TIGER 2025 ROADS whose FULLNAME is the target interstate,
                localized to the neighborhood by an anchor radius.
  * HOLC-D    = the redlined (grade-D) polygons the highway cuts through.
  * Omega     = dissolve of those HOLC-D polygons -> the study boundary.
  * barrier B = buffer(local highway centerline, row_width/2) -> the ROW the
                interstate took (used by Regime-2 severance constraints & the
                Regime-3 restoration corridor).
  * modern    = present-day TIGER roads clipped to Omega (+margin), highway
                segments tagged -> the Regime-3 base network.

Outputs:
  data/pathfinder/boundaries/{slug}.gpkg   layers: omega, barrier, holc_d,
                                            highway, modern_roads
  results/pathfinder/fig_{slug}_base.png    per-city verification map
  results/pathfinder/fig_all_base.png       2x3 overview
  data/pathfinder/boundaries/inventory.csv  areas / counts / highway length

Usage:  python pathfinder/00_boundaries.py [--cities detroit miami ...]
"""
from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import matplotlib
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pf_style as S
from pf_common import (BND_DIR, CITIES, CITY_SLUGS, HOLC_GPKG, LOCALIZE_R,
                       MODERN_MARGIN, RES, SELECT_BUF, TIGER_ROADS, holc_d_for)

warnings.filterwarnings("ignore")

HWY_CLIP_MARGIN = 500.0    # keep the highway ROW within this of Omega for barrier B


def _anchor_pt(cfg, utm):
    lon, lat = cfg["anchor"]
    return gpd.GeoSeries([Point(lon, lat)], crs=4269).to_crs(utm).iloc[0]


def build_city(slug, holc_all):
    cfg = CITIES[slug]
    utm = cfg["utm"]
    anchor = _anchor_pt(cfg, utm)

    # --- modern roads + target interstate (county TIGER, projected) ---------
    roads = gpd.read_file(TIGER_ROADS / f"tl_2025_{cfg['county']}_roads.zip").to_crs(utm)
    hwy = roads[roads["FULLNAME"].isin(cfg["fullnames"])].copy()
    # the interstate spans the whole county; clip it to a disk around the
    # neighborhood anchor so barrier B is the LOCAL ROW, not a county-long ribbon
    # (whole TIGER segments extend far past the anchor even when they pass near it).
    disk = anchor.buffer(LOCALIZE_R)
    hwy_near = hwy[hwy.geometry.distance(anchor) < LOCALIZE_R]
    hwy_disk = (unary_union(hwy_near.geometry.values).intersection(disk)
                if len(hwy_near) else unary_union(hwy.geometry.values))

    # --- HOLC-D polygons this highway cut through, near the anchor ----------
    d = holc_d_for(holc_all, cfg).to_crs(utm).copy()
    d["geometry"] = d.geometry.buffer(0)
    corridor = hwy_disk.buffer(SELECT_BUF)
    near = d[d.geometry.distance(anchor) < LOCALIZE_R]
    sel = near[near.geometry.intersects(corridor)]
    if sel.empty:                       # highway grazes but doesn't overlap a D poly
        sel = near
    omega = unary_union(sel.geometry.values).buffer(0)

    # --- barrier B: the ROW where the highway crosses the neighborhood ------
    hwy_omega = hwy_disk.intersection(omega.buffer(HWY_CLIP_MARGIN))
    if hwy_omega.is_empty:
        hwy_omega = hwy_disk
    barrier = hwy_omega.buffer(cfg["row_width"] / 2.0)
    hwy_local = gpd.GeoDataFrame(geometry=[hwy_omega], crs=utm)

    # --- modern network clipped to the study area ---------------------------
    modern = gpd.clip(roads, gpd.GeoSeries([omega.buffer(MODERN_MARGIN)], crs=utm).iloc[0])
    modern = modern[~modern.geometry.is_empty & modern.geometry.notna()].copy()
    modern["is_hwy"] = (modern["FULLNAME"].isin(cfg["fullnames"])
                        | modern["MTFCC"].eq("S1100"))

    # --- persist ------------------------------------------------------------
    gpkg = BND_DIR / f"{slug}.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    meta = dict(slug=slug, city=cfg["city"], neighborhood=cfg["neighborhood"],
                highway=cfg["highway"], build_start=cfg["build_start"])
    gpd.GeoDataFrame([{**meta, "geometry": omega}], crs=utm).to_file(gpkg, layer="omega")
    gpd.GeoDataFrame([{**meta, "row_width": cfg["row_width"], "geometry": barrier}],
                     crs=utm).to_file(gpkg, layer="barrier")
    sel.to_file(gpkg, layer="holc_d")
    hwy_local.to_file(gpkg, layer="highway")
    modern.to_file(gpkg, layer="modern_roads")

    row = dict(
        slug=slug, city=cfg["city"], neighborhood=cfg["neighborhood"],
        highway=cfg["highway"], build_start=cfg["build_start"],
        omega_km2=round(omega.area / 1e6, 3),
        n_holc_d=int(len(sel)),
        hwy_len_km=round(sum(g.length for g in hwy_local.geometry) / 1000.0, 2),
        barrier_km2=round(barrier.area / 1e6, 3),
        n_modern_edges=int(len(modern)),
        modern_local_edges=int((~modern["is_hwy"]).sum()),
        utm=utm, project_cost=cfg["project_cost"],
    )
    return row, dict(slug=slug, cfg=cfg, omega=omega, barrier=barrier, sel=sel,
                     hwy=hwy_local, modern=modern, anchor=anchor)


def _draw(ax, art, show_legend=True):
    cfg = art["cfg"]
    utm = cfg["utm"]
    m = art["modern"]
    m[~m["is_hwy"]].plot(ax=ax, color=S.C["base"], linewidth=0.5, zorder=2)
    if len(art["sel"]):
        art["sel"].plot(ax=ax, facecolor=S.C["holc_d"], alpha=0.14,
                        edgecolor=S.C["holc_d"], linewidth=0.5, zorder=1)
    gpd.GeoSeries([art["omega"]], crs=utm).plot(ax=ax, facecolor="none",
                                                edgecolor=S.C["omega"], linewidth=1.6, zorder=5)
    gpd.GeoSeries([art["barrier"]], crs=utm).plot(ax=ax, facecolor=S.C["barrier"],
                                                  alpha=0.35, edgecolor="none", zorder=3)
    art["hwy"].plot(ax=ax, color=S.C["highway"], linewidth=2.0, zorder=4)
    ax.plot(art["anchor"].x, art["anchor"].y, **S.star(), zorder=6)
    S.title(ax, f"{cfg['city']} — {cfg['neighborhood']}\n{cfg['highway']} · "
                f"Ω {art['omega'].area/1e6:.2f} km² · {len(art['sel'])} HOLC-D", fontsize=9)
    if show_legend:
        S.legend(ax, [
            ("fill", S.C["holc_d"], "redlined (HOLC-D)"),
            ("line", S.C["highway"], "interstate"),
            ("band", S.C["barrier"], "highway ROW (barrier)"),
            ("thin", S.C["base"], "street grid"),
            ("star", S.C["anchor"], "neighborhood center"),
        ])
    minx, miny, maxx, maxy = art["omega"].bounds
    mgn = 0.05 * max(maxx - minx, maxy - miny) + 150
    ax.set_xlim(minx - mgn, maxx + mgn)
    ax.set_ylim(miny - mgn, maxy + mgn)
    ax.set_aspect("equal")
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=CITY_SLUGS)
    args = ap.parse_args()

    print("reading national HOLC polygons ...", flush=True)
    holc_all = gpd.read_file(HOLC_GPKG)

    rows, arts = [], []
    for slug in args.cities:
        print(f"[{slug}]", flush=True)
        row, art = build_city(slug, holc_all)
        rows.append(row)
        arts.append(art)
        print(f"    Ω={row['omega_km2']}km²  HOLC-D={row['n_holc_d']}  "
              f"hwy={row['hwy_len_km']}km  modern_edges={row['n_modern_edges']} "
              f"({row['modern_local_edges']} local)", flush=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        _draw(ax, art)
        fig.tight_layout()
        fig.savefig(RES / f"fig_{slug}_base.png", dpi=130)
        plt.close(fig)

    inv = pd.DataFrame(rows)
    inv.to_csv(BND_DIR / "inventory.csv", index=False)
    print("\n" + inv.to_string(index=False), flush=True)

    if len(arts) > 1:
        n = len(arts)
        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        for i, (ax, art) in enumerate(zip(axes.ravel(), arts)):
            _draw(ax, art, show_legend=(i == 0))
        for ax in axes.ravel()[n:]:
            ax.set_axis_off()
        fig.suptitle("PathFinder — severed-neighborhood study areas (Ω) + highway barrier B",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(RES / "fig_all_base.png", dpi=130)
        plt.close(fig)
    print(f"\nsaved -> {BND_DIR}/  and  {RES}/", flush=True)


if __name__ == "__main__":
    main()
