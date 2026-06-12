# UOI Pipeline — Stages 1–3

Implements the first three stages of the research design: street-network
extraction, the Urban Optionality Index (UOI), and stratified sampling of
deep-analysis tracts for the RJ-MCMC sampler.

```
01_extract_networks.py   TIGER tracts + OSMnx walk networks -> data/graphs/{GEOID}.graphml
02_compute_uoi.py        4 UOI dimensions + morphology features -> data/outputs/uoi_metrics.parquet
03_stratified_sample.py  typology strata, Pareto frontier, N-tract sample -> data/outputs/sample_tracts.csv
```

## Quick start (pilot: San Francisco County)

```bash
python 01_extract_networks.py --state 06 --county 075
python 02_compute_uoi.py
python 03_stratified_sample.py --n 50
```

All stages are resumable: re-running skips work that is already done.

## UOI definitions (all oriented higher = better)

| Dimension     | Metric                                                            |
|---------------|-------------------------------------------------------------------|
| connectivity  | link-node ratio (undirected edges / nodes)                        |
| efficiency    | 1 / average fractional circuity (length-weighted, edge-level)     |
| accessibility | mean nodes reachable within an 800 m network walk ("reach")       |
| equity        | 1 − Gini of the per-node reach distribution                       |

UOI is **not** collapsed into one score — stage 3 flags the Pareto frontier
across the four dimensions.

Accessibility currently uses network nodes as the opportunity proxy. To use
real opportunities, swap in LEHD LODES workplace-area jobs per block, snapped
to nearest nodes (hook: `tract_metrics()` in `02_compute_uoi.py`).

## Typology stratification (stage 3)

KMeans (k=4) on standardized [orientation entropy, dead-end fraction,
circuity, intersection density]; clusters auto-labelled gridded / cul_de_sac /
organic / hybrid by their feature profiles. Sample is allocated to strata
proportionally (≥1 each). Inspect `typology_assignments.csv` to sanity-check
the labels before trusting the sample.

## Scaling to all 84,414 tracts

- Stage 1 downloads **one Overpass query per county** (~3,200 for the U.S.),
  not per tract. Still, Overpass etiquette makes the full U.S. a multi-day,
  rate-limited job. The robust path at full scale is **Geofabrik state .pbf
  extracts + pyrosm/osmium** to build state graphs locally, keeping the same
  per-tract truncation step — only `extract_county()`'s download line changes.
- Stage 2 is embarrassingly parallel over tracts (~1–5 s per tract → roughly
  one to several CPU-days for the U.S.; trivially distributed with
  `--limit`/sharding or multiprocessing).
- Stage 3 runs in seconds at any scale.

## Known modelling choices to revisit

- 800 m reach cutoff and node-as-opportunity proxy (LODES upgrade above).
- Edge-level circuity (cheap) vs. sampled origin–destination circuity
  (closer to "trip" efficiency, ~10× cost).
- Equity measured within-tract; the design doc's equity *analysis* (HOLC
  comparison, Theil decomposition) is a later stage and unaffected.
- The design doc's "192 × 5 = 1,000" sample-size justification needs fixing
  (it is 960); stratum-proportional allocation here just takes `--n`.
