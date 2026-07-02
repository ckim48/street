---
title: "CK_street — UOI 파이프라인 메소드 & 주요 수식 정리"
date: 2026-07-02
---

# 전체 파이프라인 개요

| 단계 | 스크립트 | 내용 |
|------|----------|------|
| 01 | `01_extract_networks*.py` | TIGER tract 경계 + OSM 보행 네트워크 추출 → GraphML |
| 02 | `02_compute_uoi_spec.py` | tract별 UOI 6개 지표 계산 |
| 03 | `viz_top1000.py` | 백분위 기반 composite 점수, top-1000 tract 선정 |
| 04/05 | `04_sampler.py`, `05_mcmc_spec.py` | RJ-MCMC 반사실(counterfactual) 네트워크 탐색 → distance-to-frontier |
| 06 | `06_synthesize.py` | 벤치마크 기반 가상 네트워크 합성 |
| 07 | `07_gnn_surrogate.py` | GNN 서로게이트로 전국 ~84k tract에 dtf 예측 확장 |
| 08 | `08b`–`08e` | 사회경제 지표(ACS, Opportunity Atlas, FARS 등)와의 상관·회귀 분석 |

# 1. UOI 지표 6개 (Stage 2)

Tract별 무향 투영 그래프 $G=(V,E)$, $n=|V|$, $m=|E|$에서 계산한다.

**1) Link-node ratio** (높을수록 좋음, 권장 $\ge 1.4$):
$$\mathrm{LNR} = \frac{m}{n}$$

**2) Connected node ratio** (높을수록 좋음, 권장 $\ge 0.7$):
$$\mathrm{CNR} = \frac{n_{\deg\ge 3}}{n_{\deg\ge 3} + n_{\deg=1}}$$

**3) Intersection density** (높을수록 좋음, 권장 $> 140/\mathrm{mi}^2$):
$$\mathrm{ID} = \frac{n_{\deg\ge 3}}{\mathrm{ALAND}\ (\mathrm{mile}^2)}$$

**4) Median block length** (낮을수록 좋음, 권장 $\le 600\,\mathrm{ft}$):
$$\mathrm{MBL} = \operatorname{median}_e(\ell_e) \times 3.281\ \mathrm{ft}$$

**5) Walking circuity** (권장 밴드 $[1.2, 1.7]$) — OD 노드쌍 500개 샘플:
$$C = \mathbb{E}_{(s,t)}\!\left[\frac{d_{\mathrm{net}}(s,t)}{d_{\mathrm{euc}}(s,t)}\right]$$

**6) Pedshed reach** (높을수록 좋음) — tract 내부 H3 res-9 격자점 $p$마다 최근접 노드로 스냅 후, 네트워크 거리 400 m ego-graph 내 도로 총연장을 반경 400 m 원판 면적으로 정규화:
$$R = \mathbb{E}_{p \in \mathrm{H3}}\!\left[\frac{\sum_{e \in \mathrm{ego}(p,\,400\mathrm{m})} \ell_e}{\pi \cdot 400^2}\right]$$

# 2. Composite UOI score (Stage 3)

각 지표를 전국 백분위 순위로 변환하고(방향 정렬: block length와 circuity는 반전), 6개 백분위의 평균을 취한다:
$$\mathrm{UOI\_score} = \frac{1}{6}\sum_{i=1}^{6} \mathrm{pct}_i \in [0,1]$$

이 점수로 전국 tract를 순위화하여 top-1000을 선정한다. (구버전 `02_compute_uoi.py`의 4차원 — connectivity, efficiency, accessibility, equity $=1-\mathrm{Gini}(\mathrm{reach})$ — 은 단일 점수로 합치지 않고 Pareto frontier만 표시한다.)

# 3. RJ-MCMC 반사실 네트워크 탐색 (Stage 4–5)

**상태공간.** tract 폴리곤 내부의 물리적으로 타당한 평면 그래프. 엣지는 직선으로 추상화하며, 실제 네트워크도 같은 추상화로 재채점하여 비교의 공정성을 유지한다.

**제약.** 평면성(엣지 교차 금지), 연결성, $\deg \le 5$, 노드 최소 간격 15 m, 엣지 길이 20–250 m, 노드·엣지 수는 초기 네트워크의 3배 이하.

**이동(moves).** shift(0.40), add_edge(0.15), remove_edge(0.15), add_node/subdivide(0.15), remove_node/merge(0.15). 각 이동은 역이동과 짝을 이루며 Hastings 보정을 명시적으로 계산한다. 예: add_edge의

$$\log H = \log\frac{q_{\mathrm{rev}}}{q_{\mathrm{fwd}}}, \qquad q_{\mathrm{fwd}} = \frac{1}{n\,c_u} + \frac{1}{n\,c_v}, \qquad q_{\mathrm{rev}} = \frac{1}{|\mathrm{removable}(G')|}$$

(add/remove_node는 반지름 20 m 균등 디스크 제안밀도 $1/(\pi r^2)$ 항 포함.)

**목표분포와 수락 규칙.** sharpness $S=60$, 온도 사다리 $\beta_t$에 대해
$$\pi(G) \propto \exp\big(S\,\beta\,E(G)\big), \qquad \text{accept if } \log u < S\,\beta_t\,(E_2 - E_1) + \log H$$

**에너지 (6지표 spec 버전).** 개선 벡터의 가중합:
$$E(G) = \sum_{i=1}^{6} w_i \tanh\!\left(\frac{x_i}{\tau}\right), \qquad w \sim \mathrm{Dirichlet}(\mathbf{1}_6), \qquad \tau = 0.5$$

$x_i$는 실제 네트워크 대비 방향 정렬된 스케일-프리 로그 개선량이다:

- 높을수록 좋은 지표(1, 2, 3, 6): $x_i = \log(v_i / v_i^{\mathrm{real}})$
- Block length(4): $x_4 = \log(v_4^{\mathrm{real}} / v_4)$
- Circuity(5): 밴드 위반 페널티의 감소량 $x_5 = P(c^{\mathrm{real}}) - P(c)$, 여기서

$$P(c) = \begin{cases} \log(1.2/c) & c < 1.2 \\ 0 & 1.2 \le c \le 1.7 \\ \log(c/1.7) & c > 1.7 \end{cases}$$

$\tanh$ 포화로 인해 어느 한 지표가 에너지를 독식할 수 없으므로, 최적해는 "모든 지표를 권장 기준 쪽으로 끌어올리는" 네트워크가 된다. (구버전 4차원 에너지는 $E = \sum_i w_i \ln(u_i/u_i^{\mathrm{real}})$.)

**MCMC-시간 서로게이트 평가.** 지표 1–4는 후보 그래프에서 $O(n+m)$ 정확 계산. 지표 5–6은 고정 anchor 12개(최근접 노드 스냅)에서 한 번의 multi-source Dijkstra로 circuity(anchor 쌍 네트워크/직선 거리)와 pedshed(anchor별 400 m 도달 도로연장)를 동시 산출한다.

**Parallel tempering.** $\beta \in \mathrm{geomspace}(1.0,\ 0.18,\ T)$, 20 iteration마다 인접 온도 상태 스왑:
$$\log u < S\,(\beta_t - \beta_{t+1})(E_{t+1} - E_t)$$

**수렴 진단 — split Gelman–Rubin.** 동일 가중치 $w$의 replica trace 후반부를 반씩 분할하여
$$\hat R = \sqrt{\frac{\frac{L-1}{L} W + \frac{B}{L}}{W}}$$
($W$: 체인 내 분산 평균, $B$: 체인 간 분산.)

**핵심 산출량 — distance-to-frontier (dtf).** posterior 개선 벡터 구름과 실제 네트워크(개선 공간의 원점 $\mathbf{0}$)를 합친 집합의 Pareto front에 대해, 참조점 $\mathbf{-1}$($\tanh$ 하한) 기준 하이퍼볼륨의 상대 부족분을 Monte Carlo로 추정한다:
$$\mathrm{dtf} = 1 - \frac{HV(\mathrm{real})}{HV(\mathrm{front})} \in [0, 1]$$

$\mathrm{dtf}=0$이면 실제 네트워크가 이미 Pareto 최적이고, 값이 클수록 같은 제약 하에서 개선 여지가 크다.

# 4. 가상 네트워크 합성 (Stage 6)

실제 참조 네트워크가 없으므로, top-1000 tract의 지표 중앙값을 target으로 하는 "reach-or-better" 보상을 쓴다. 차원별 보상 $r_i \in (-1, 0]$은 목표 도달·초과 시 0에서 plateau하며,
$$E = \sum_{i=1}^{6} w_i\, r_i, \qquad w \sim \mathrm{Dirichlet}(\mathbf{1}_6)$$
최대값 $E=0$은 "모든 지표가 top-1000 벤치마크 달성"을 뜻한다. 시드 아키타입은 gridded / organic(Delaunay) / hybrid 3종.

# 5. GNN 서로게이트 (Stage 7)

MCMC로 라벨링된 1,000개 tract에서 그래프 수준 회귀 $G \mapsto \mathrm{dtf}$를 학습하여, 전국 ~84k tract를 forward pass(~ms/tract)로 채점한다.

- 노드 피처: $[\deg,\ \mathbb{1}(\deg\ge3),\ \mathbb{1}(\deg=1),\ x_{\mathrm{norm}},\ y_{\mathrm{norm}}]$, 엣지 피처: 정규화 길이
- 모델: 3× GraphSAGE → global mean|max pooling → MLP → scalar
- 주의: dtf 라벨이 미수렴 MCMC 산출($\hat R$ 중앙값 ~1.6)이라 노이즈가 있으며, 서로게이트의 달성 가능 $R^2$은 라벨 품질에 의해 상한이 정해진다. 순위상관을 병행 보고한다.

# 6. 사회경제 상관·회귀 분석 (Stage 8)

**조정 회귀 (8d).** 밀도·소득·인종 교란과 주(state) 고정효과를 통제한 표준화 OLS:
$$z(Y) = \beta_0 + \beta_{\mathrm{UOI}}\, z(\mathrm{UOI}) + \beta_1\, z(\log_{10}\rho_{\mathrm{pop}}) + \beta_2\, z(\log_{10}\mathrm{income}) + \beta_3\, z(\mathrm{pct\_white}) + \mathrm{state\ FE} + \varepsilon$$

$$\hat\beta = (X^\top X)^{-1} X^\top y, \qquad \mathrm{SE}(\hat\beta_j) = \sqrt{\hat\sigma^2 \left[(X^\top X)^{-1}\right]_{jj}}$$

원시 Spearman $\rho$와 조정된 $\beta_{\mathrm{UOI}}$를 나란히 보고한다. 아웃컴은 0.5/99.5 백분위로 winsorize하고, 인구 200명 미만 tract의 per-capita 보행자 사망률은 제외한다.

**보행자 안전 심층 분석 (8e).** 조정 후에도 남은 UOI ↑ → 보행자 사망률 ↑ ($\beta \approx +0.21$) 관계의 원인을 세 갈래로 추적한다: (1) 노출 검정 — ACS walk-to-work share를 추가 통제했을 때 효과가 소멸하는지, (2) 구성지표 분해 — 6개 지표 중 어느 것이 위험을 담지하는지, (3) 메커니즘 — FARS 보행자 사망 사고를 tract에 결합해 UOI 5분위별로 도로 기능등급, 도시/농촌, 교차로 vs mid-block을 프로파일링.
