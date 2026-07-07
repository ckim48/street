# Tier 1 — HOLC boundary RD deep dive (6 cities)

City-level regression-discontinuity (RD) experiment that sits **beside** the
national Tier 0 UOI result. For six HOLC-mapped cities it estimates a geographic
discontinuity in street-network organization (the six-metric UOI / **Oi** index)
across HOLC grade boundaries, plus the mechanism narratives and visual
validation panels from the *Tier 1 Experiment Guide*.

> **Role of Tier 1.** Mechanisms, validation, and visuals — *not* a replacement
> for the Tier 0 headline estimate. City estimates are illustrative/diagnostic.

## Cities and roles

| slug | city | HOLC yr | CBSA | role | watch-out |
|------|------|--------:|-----:|------|-----------|
| chicago | Chicago, IL | 1940 | 16980 | Benchmark; large C-D stock on a uniform grid | drop lake & river segments |
| philadelphia | Philadelphia, PA | 1937 | 37980 | Uniform rowhouse grid; strong continuity | flag rail-line boundaries |
| baltimore | Baltimore, MD | 1937 | 12580 | Rowhouse fabric; redlining-lit overlap | flag harbor-adjacent segments |
| detroit | Detroit, MI | 1939 | 19820 | Barrier-robustness; post-1950 divergence | robustness, not headline |
| atlanta | Atlanta, GA | 1938 | 12060 | Sunbelt growth; widening-gap trajectory | topography, irregular fabric |
| los_angeles | Los Angeles, CA | 1939 | 31080 | Freeway-mechanism chapter | treat freeways as post-treatment |

## Pipeline (maps to the guide's steps)

| script | guide step | what it does |
|--------|-----------|--------------|
| `00_fetch_tier1.sh` | 1 (data) | fetch HOLC polygons, Markley ADS, HISDAC FBUY, CHRONEX-US |
| `01_holc_boundaries.py` | 1–2 | subset city, dissolve grades, extract & **segment** C-D / B-C frontiers, classify barriers (street/rail/water/freeway/harbor) |
| `02_decade_graphs.py` | 3.1 | build 1940…2020 road graphs per city from CHRONEX-US dated edges (OSM present-day fallback) |
| `03_compute_oi.py` | 3.2 | local six-metric Oi on both sides of each boundary at a grid of signed distances, per decade |
| `04_rd_estimate.py` | 5 | local-linear geographic RD (signed distance, lower-grade treatment, boundary-segment FE, cluster-robust SE) |
| `05_figures.py` | 6 | boundary/barrier maps, Oi-by-decade τ trajectories, cross-city RD forest |

Shared code: `tier1_common.py` (paths, city table, constants), `oi_local.py`
(graph-from-lines + the six-metric local Oi, identical definitions to
`02_compute_uoi_spec.py`).

Run everything: `./tier1/run_tier1.sh` (see flags inside). Individual stages take
`--cities`.

## RD design

- **Running variable** `x` = signed perpendicular distance to the HOLC frontier;
  `+` on the lower-grade (worse-rated) side.
- **Treatment** `T = 1{x>0}` = the lower-grade side (Guide 5.2).
- **Model** `y = a + τ·T + b₁x + b₂(T·x) + segment-FE`, triangular kernel within
  bandwidth `h`, boundary-segment fixed effects absorbed by a weighted
  within-transform, **cluster-robust (by segment) SEs**. `τ` is the jump in the
  network metric crossing from higher to lower grade.
- Two boundary contrasts, reported separately: **C-D** and **B-C**.
- Clean sample = segments classified `street`; `rail`/`water`/`freeway`/`harbor`
  segments are excluded by default (barrier-robustness = re-run with `--barriers all`).

## Data provenance (URLs verified 2026-07-04)

| dataset | source | file | key fields |
|---------|--------|------|-----------|
| HOLC polygons | Mapping Inequality (U. Richmond) | `mappinginequality.gpkg` | `area_id`, `city`, `state`, `grade`, `label` |
| ADS covariates | Markley, OSF `qytj8` | `ADS_FINAL.csv` | join `CITY`+`HOLC_ID` ↔ polygon `label` |
| First built-up year | HISDAC-US, Harvard Dataverse `10.7910/DVN/HHFM5E` | `FBUY.tif` (EPSG:5070, 250 m) | pixel = first built-up year; `<=1940` mask |
| dated road network | CHRONEX-US, Figshare 28644674 | `chronex_us_<cbsa>.gpkg` (per-CBSA, UTM) | `yr_upper_M1` (yr), `MTFCC_CODE` |

## Modeling notes / caveats

- **CHRONEX year field.** Default `yr_upper_M1` (authors' baseline). M1 tends to
  estimate roads as *too old*, so within the historic HOLC footprint the decade
  graphs are near-identical (the core was largely built by 1940) — this is
  expected and is why Chicago/Philadelphia/Baltimore read as *continuity* cases
  while growth signal concentrates in Atlanta/LA. Switch with
  `02_decade_graphs.py --year-field yr_lower_M1` (or `yr_upper_M2`) for a
  robustness pass; M2/M3 give younger but more disconnected historical nets.
- Highways/ramps (`MTFCC_CODE` S1100/S1630) are dropped so the graph is the local
  street fabric the Oi index measures; freeways re-enter only as *barriers* (LA
  mechanism chapter) via the boundary classification, not as edges.
- HISDAC FBUY is fetched for the Step-4 "both sides built-up by 1940" check; the
  join into the RD sample is a TODO hook (see `08b`-style raster sampling).
- topoView / Sanborn same-corner validation (Guide Step 4) is manual; keep the
  city-log fields in `<slug>_inventory.csv` + boundary gpkg as the corridor index.

## Outputs

- `data/tier1/boundaries/{slug}_grades.gpkg`, `{slug}_boundaries.gpkg`, `{slug}_inventory.csv`
- `data/tier1/graphs/{slug}_{decade}.graphml`
- `data/tier1/oi/{slug}_oi.parquet`
- `data/tier1/rd/{slug}_rd.csv`, `results/tier1/rd_all.csv`
- `results/tier1/fig_{slug}_boundary_map.png`, `fig_{slug}_oi_by_decade.png`, `fig_tier1_rd_summary.png`
