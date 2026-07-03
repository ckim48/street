# Overall Pipeline Overview

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Stage               Script                               Content
  ------------------- ------------------------------------ -----------------------------------------------------------------------------------------------------------
  01                  `01_extract_networks*.py`            Extract TIGER tract boundaries + OSM pedestrian network → GraphML

  02                  `02_compute_uoi_spec.py`             Compute the six UOI metrics per tract

  03                  `viz_top1000.py`                     Percentile-based composite score; selection of the top-1000 tracts

  04/05               `04_sampler.py`, `05_mcmc_spec.py`   RJ-MCMC counterfactual network search → distance-to-frontier

  06                  `06_synthesize.py`                   Benchmark-based synthesis of hypothetical networks

  07                  `07_gnn_surrogate.py`                GNN surrogate extends dtf prediction to \~84k tracts nationwide

  08                  `08b`–`08e`                          Correlation and regression analysis against socioeconomic indicators (ACS, Opportunity Atlas, FARS, etc.)
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. Data Extraction and Pre-processing (Stage 1)

## 1.1 Data Sources

- **Census TIGER/Line (2024 vintage)** — census-tract boundary shapefiles are downloaded per state, providing the 11-digit GEOID, land area (ALAND), water area (AWATER), and polygon geometry (EPSG:4326). Tract attributes are saved to `tracts_{state}.gpkg` and reused by all later stages (area normalization, the MCMC polygon constraint, and panel joins).
- **OpenStreetMap pedestrian network** — two extraction paths:
  - *Overpass API* (pilot): one query per county, split locally afterwards. This keeps API load at roughly 3,200 county queries nationwide instead of 84,414 per-tract queries.
  - *Geofabrik state `.pbf` extracts* (national scale, default): one `.pbf` per state, parsed locally with pyrosm. Because a whole-state walk graph for a large state exceeds 7M nodes and runs out of memory, each state is processed **county by county**, parsing only that county's bounding box to bound peak memory.
- **External socioeconomic data** (joined in Stage 8): Opportunity Atlas (economic mobility, incarceration), Eviction Lab (eviction filing rates, 2014–18 mean), FARS (pedestrian fatalities 2017–21, spatially joined by crash coordinates), LODES8 (job counts, stable-job share), and ACS 5-year (population, income, race, education, walk-to-work share).

## 1.2 Network Extraction Procedure

1. **County polygon construction**: the union of a county's tract polygons is projected to UTM and buffered by **300 m** — this prevents streets crossing the county boundary from being clipped, which would distort the metrics of boundary tracts.
2. **Walk-graph construction**: the OSMnx `network_type="walk"` filter keeps only walkable ways (`retain_all=True`, `truncate_by_edge=True`); OSM boolean tags (`oneway`, `reversed`) are normalized, and the graph is **topologically simplified** (`simplify_graph`) to intersection-to-intersection segments.
3. **Tract splitting**: the county graph is cut into per-tract subgraphs using the truncate-by-edge rule — nodes inside the polygon plus their direct neighbors are retained, so edges crossing the tract boundary survive. A county-wide spatial index (R-tree) is built once, so the per-tract cost scales with the number of nearby nodes rather than the size of the whole graph. The result is one `{GEOID}.graphml` per tract.
4. **Robustness**: every step is resumable (already-extracted tracts are skipped); tracts without streets (water, parks) and county-level failures are recorded in `extract_log.csv` without aborting the run. Mega-counties are isolated in separate processes so an out-of-memory kill cannot block the rest of the state.

## 1.3 Pre-processing (Common Steps Before Analysis)

- **For metric computation (Stage 2)**: graphs are projected and converted to undirected form. Intersections ($deg \geq 3$) and dead ends ($deg = 1$) are classified by node degree, and tracts with fewer than 5 nodes are excluded as `too_small`.
- **For MCMC state initialization (Stages 4–5)**: graphs are converted to projected undirected **simple graphs** — self-loops removed, only the largest connected component kept, curved geometry abstracted to straight edges with lengths recomputed as Euclidean distances (the real network is re-scored under the same abstraction to keep the comparison fair). The tract polygon, buffered by 20 m, serves as the node-placement constraint region.
- **For panel construction (Stage 8b)**: all external data are joined on the standardized 11-digit GEOID (FARS crashes are spatially joined to tract polygons by coordinates). ALAND is converted to km² as the denominator for population and job densities, and pilot/duplicate layers are dropped.

# 2. The Six UOI Metrics (Stage 2)

For each tract, the metrics are computed on the undirected projected graph $G = (V,E)$, $n = |V|$, $m = |E|$.

**1) Link-node ratio** (higher is better; recommended $\geq 1.4$):

$$LNR = \frac{m}{n}$$

**2) Connected node ratio** (higher is better; recommended $\geq 0.7$):

$$CNR = \frac{n_{deg \geq 3}}{n_{deg \geq 3} + n_{deg = 1}}$$

**3) Intersection density** (higher is better; recommended $> 140/{mi}^{2}$):

$$ID = \frac{n_{deg \geq 3}}{ALAND\ ({mile}^{2})}$$

**4) Median block length** (lower is better; recommended $\leq 600\, ft$):

$$MBL = {median}_{e}(\ell_{e}) \times 3.281\ ft$$

**5) Walking circuity** (recommended band $\lbrack 1.2,1.7\rbrack$) — sample of 500 OD node pairs:

$$C = \mathbb{E}_{(s,t)}\left\lbrack \frac{d_{net}(s,t)}{d_{euc}(s,t)} \right\rbrack$$

**6) Pedshed reach** (higher is better) — at each H3 res-9 lattice point $p$ inside the tract, snap to the nearest node; then normalize the total road length within the 400 m network-distance ego-graph by the area of a 400 m-radius disk:

$$R = \mathbb{E}_{p \in H3}\left\lbrack \frac{\sum_{e \in ego(p,\, 400m)}^{}\ell_{e}}{\pi \cdot 400^{2}} \right\rbrack$$

# 3. Composite UOI Score (Stage 3)

Each metric is converted to a nationwide percentile rank (with directional alignment: block length and circuity are inverted), and the mean of the six percentiles is taken:

$$UOI\_ score = \frac{1}{6}\sum_{i = 1}^{6}{pct}_{i} \in \lbrack 0,1\rbrack$$

These scores rank all tracts nationwide to select the top-1000. (The four dimensions of the legacy `02_compute_uoi.py`—connectivity, efficiency, accessibility, equity$= 1 - Gini(reach)$—are not merged into a single score but are shown only as a Pareto frontier.)

# 4. RJ-MCMC Counterfactual Network Search (Stages 4–5)

**State space.** A physically plausible planar graph inside the tract polygon. Edges are abstracted as straight lines, and the real network is re-scored under the same abstraction to keep the comparison fair.

**Constraints.** Planarity (no edge crossings), connectivity, $deg \leq 5$, minimum node spacing of 15 m, edge length 20–250 m, and node/edge counts no more than three times those of the initial network.

**Moves.** shift(0.40), add_edge(0.15), remove_edge(0.15), add_node/subdivide(0.15), remove_node/merge(0.15). Each move is paired with its reverse move, and the Hastings correction is computed explicitly. For example, for add_edge:

$$\log H = log\frac{q_{rev}}{q_{fwd}},\quad\quad q_{fwd} = \frac{1}{n\, c_{u}} + \frac{1}{n\, c_{v}},\quad\quad q_{rev} = \frac{1}{|removable(G')|}$$

(add/remove_node include a uniform-disk proposal-density term of radius 20 m, $1/(\pi r^{2})$.)

**Target distribution and acceptance rule.** With sharpness $S = 60$ and temperature ladder $\beta_{t}$, the acceptance rule is:

$$\pi(G) \propto exp\left( S\,\beta\, E(G) \right),\quad\quad\text{accept if }\log u < S\,\beta_{t}\,(E_{2} - E_{1}) + logH$$

**Energy (six-metric spec version).** A weighted sum of the improvement vector:

$$E(G) = \sum_{i = 1}^{6}w_{i}\tanh\left( \frac{x_{i}}{\tau} \right),\quad\quad w \sim Dirichlet(\mathbf{1}_{6}),\quad\quad\tau = 0.5$$

$x_{i}$ is the direction-aligned, scale-free log improvement measured against the real network:

- Metrics where higher is better (1, 2, 3, 6): $x_{i} = log(v_{i}/v_{i}^{real})$
- Block length (4): $x_{4} = log(v_{4}^{real}/v_{4})$
- Circuity (5): the reduction in the band-violation penalty $x_{5} = P(c^{real}) - P(c)$, where

$$P(c) = \left\{ \begin{matrix}
log(1.2/c) & c < 1.2 \\
0 & 1.2 \leq c \leq 1.7 \\
log(c/1.7) & c > 1.7
\end{matrix} \right.\ $$

Because $\tanh$ saturation prevents any single metric from monopolizing the energy, the optimum is a network that “pulls every metric toward its recommended criterion.” (The legacy four-dimensional energy is $E = \sum_{i}^{}w_{i}ln(u_{i}/u_{i}^{real})$.)

**MCMC-time surrogate evaluation.** Metrics 1–4 are computed exactly in $O(n + m)$ on the candidate graph. Metrics 5–6 are obtained simultaneously from a single multi-source Dijkstra over 12 fixed anchors (snapped to nearest nodes), yielding circuity (network/straight-line distance for anchor pairs) and pedshed (400 m reachable road length per anchor).

**Parallel tempering.** $\beta \in geomspace(1.0,\ 0.18,\ T)$, with adjacent-temperature states swapped every 20 iterations:

$$\log u < S\,(\beta_{t} - \beta_{t + 1})(E_{t + 1} - E_{t})$$

**Convergence diagnostic — split Gelman–Rubin.** With equal weights $w$, the latter half of each replica trace is split into two halves, giving

$$\widehat{R} = \sqrt{\frac{\frac{L - 1}{L}W + \frac{B}{L}}{W}}$$

($W$: mean within-chain variance, $B$: between-chain variance.)

**Key output — distance-to-frontier (dtf).** For the Pareto front of the set that combines the posterior improvement-vector cloud with the real network (the origin $\mathbf{0}$ of the improvement space), and relative to the reference point $\mathbf{- 1}$ ($\tanh$ lower bound), the relative shortfall in hypervolume is estimated by Monte Carlo:

$$dtf = 1 - \frac{HV(real)}{HV(front)} \in \lbrack 0,1\rbrack$$

$dtf = 0$ means the real network is already Pareto-optimal; larger values indicate greater room for improvement under the same constraints.

# 5. Synthesis of Hypothetical Networks (Stage 6)

Because no real reference network exists, a “reach-or-better” reward targeting the median metric values of the top-1000 tracts is used. The per-dimension reward $r_{i} \in ( - 1,0\rbrack$ plateaus at 0 once the target is reached or exceeded, and

$$E = \sum_{i = 1}^{6}w_{i}\, r_{i},\quad\quad w \sim Dirichlet(\mathbf{1}_{6})$$

the maximum value $E = 0$ denotes “all metrics attain the top-1000 benchmark.” The seed archetypes are three: gridded, organic (Delaunay), and hybrid.

# 6. GNN Surrogate (Stage 7)

A graph-level regression $G \mapsto dtf$ is trained on the 1,000 MCMC-labeled tracts and scores all \~84k tracts nationwide via a forward pass (\~ms/tract).

- Node features: $\lbrack deg,\ \mathbb{1}(deg \geq 3),\ \mathbb{1}(deg = 1),\ x_{norm},\ y_{norm}\rbrack$, edge feature: normalized length
- Model: 3× GraphSAGE → global mean\|max pooling → MLP → scalar
- Note: because the dtf labels are outputs of non-converged MCMC ($\widehat{R}$ median \~1.6), they are noisy, and the surrogate’s attainable $R^{2}$ is upper-bounded by the label quality. Rank correlation is reported alongside.

# 7. Socioeconomic Correlation and Regression Analysis (Stage 8)

**Adjusted regression (8d).** A standardized OLS controlling for density, income, and race confounders and for state fixed effects:

$$z(Y) = \beta_{0} + \beta_{UOI}\, z(UOI) + \beta_{1}\, z(\log_{10}\rho_{pop}) + \beta_{2}\, z(\log_{10}income) + \beta_{3}\, z(pct\_ white) + state\ FE + \varepsilon$$

$$\widehat{\beta} = (X^{\top}X)^{- 1}X^{\top}y,\quad\quad SE({\widehat{\beta}}_{j}) = \sqrt{{\widehat{\sigma}}^{2}\left\lbrack (X^{\top}X)^{- 1} \right\rbrack_{jj}}$$

The raw Spearman $\rho$ and the adjusted $\beta_{UOI}$ are reported side by side. Outcomes are winsorized at the 0.5/99.5 percentiles, and per-capita pedestrian fatality rates for tracts with populations under 200 are excluded.

**Pedestrian-safety deep dive (8e).** The residual UOI ↑ → pedestrian-fatality ↑ ($\beta \approx + 0.21$) relationship that persists after adjustment is traced along three lines: (1) exposure test — whether the effect vanishes when ACS walk-to-work share is added as a control; (2) component decomposition — which of the six metrics carries the risk; (3) mechanism — joining FARS pedestrian-fatality crashes to tracts and profiling road functional class, urban/rural, and intersection vs. mid-block by UOI quintile.
