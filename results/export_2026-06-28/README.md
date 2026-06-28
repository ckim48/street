# UOI Street-Network Pipeline — Results Export (2026-06-28)

Consolidated figures + data from every stage of the Urban Optionality Index (UOI)
street-network pipeline. Headline numbers in `SUMMARY_STATS.json`.

The UOI uses the design-doc **6-metric spec**: link-node ratio (≥1.4),
connected-node ratio (≥0.7), intersection density (>140 /mi²), median block
length (≤600 ft), walking circuity (1.2–1.7 band), pedshed reach (higher better).

---

## 01_national_uoi — full-U.S. UOI computation
The core dataset: every metric for **84,395 tracts with metrics / 84,446 extracted, 52 states** (50 + DC + PR).
- `uoi_spec_metrics_national.parquet` / `.csv` — full table (6 metrics + bounds flags per tract)
- `per_state_summary.csv` — tract counts + median metrics per state
- `fig_national_metric_distributions.png` — 6 histograms vs. design-doc bounds
- `uoi_correlation.png`, `uoi_maps.png` — correlation heatmap + choropleth
- `uoi_spec_vs_code.csv` — spec metrics vs. repo's original 4-dim UOI

**National medians:** LNR 1.33 · CNR 0.77 · density 260/mi² · block 167 ft · circuity 1.51 · pedshed 0.0034.
Block-length is the most-satisfied bound (94.5% in-range); link-node ratio the least (31.8%).

## 02_top1000 — top-1000 tracts by UOI
Best-scoring tracts nationally (elite dense urban grids — Seattle/NYC/Chicago dominate).
- `top1000_uoi.csv`, `uoi_scores_all.csv` — ranked scores
- network galleries (top/mid/last 24), score distribution, metric profile/correlation, by-state bar.

## 03_mcmc_optimal — RJ-MCMC optimal-network search
Parallel-tempering MCMC searches the achievable UOI frontier per tract.
**distance_to_frontier (dtf)** = how far the real network sits from its own optimum (0 = already optimal).
- `dtf_table.csv` — per-tract dtf
- `fig_dtf_distribution.png`, `fig_metric_shift.png` (real→optimal per metric),
  `fig_best_networks.png`, `fig_optimal_gallery.png`
- `fig_dtf_elite_vs_national.png` — **key contrast** below
- `per_tract_networks/` — before→after per tract

**Finding:** 1,800 tracts searched (1,000 elite + 800 stratified national).
Elite real networks are already near-optimal (dtf median **0.011**); the national
sample has far more headroom (dtf median **0.62**). Improvement axes are almost
always circuity (into band) + pedshed (+15–35%); LNR/CNR/density/block are saturated.
*Caveat:* MCMC under-converged (R-hat median ~1.6, some national tracts >2);
point estimates usable, posterior diagnostics not paper-grade.

## 04_gnn_surrogate — GNN dtf predictor
GraphSAGE surrogate predicting dtf from graph + the 6 metrics (avoids running MCMC nationally).
- `fig_pred_vs_true.png` — current model · `_top1000_bak.png` — old top-1000-only model
- `gnn_dtf_predictions.csv` — national predictions (with OOD flags)

*Note:* the top-1000-only model exploded out-of-distribution on ordinary tracts;
retrained on the combined **1,800** (elite + national) sample to fix this.

## 05_virtual_synthesis — benchmark-calibrated synthetic networks
Synthesizes optimal *city-free* virtual networks (grid / organic / hybrid seeds) targeting top-1000 median metrics.
- `fig_synth_networks.png`, `synth_metrics.csv`, `run.log`

**Finding:** grid archetypes score best → validates that gridded networks are UOI-optimal.
Density + block-length cannot both be matched by a uniform grid (geometric trade-off, expected/documented).

## 06_alabama_casestudy — validated single-state deep-dive
Alabama (FIPS 01) end-to-end, with an independent UOI verification (100/100 recompute match to 1e-14).
- county/network figures, correlation, maps; metric/summary/verification tables.

---

### Coverage gaps (honest)
- **Connecticut (323 tracts, 0.4%)** — pyrosm 0.8.0 parse bug in 2024 planning regions; deferred.
- A handful of per-state water/no-network "empty" tracts (legit, not failures).
