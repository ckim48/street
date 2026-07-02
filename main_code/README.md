# main_code — 알고리즘·메소드 코드 모음

UOI street-network 파이프라인의 **방법론 핵심 코드만** 모은 폴더입니다
(레퍼런스/아카이브용 스냅샷). 데이터 다운로드·추출(01_*, 08a/08b)과 실행
드라이버(run_*.sh)는 저장소 루트에 있습니다. 방법론과 수식 정리는
`docs/METHODS_UOI.docx` 참조.

**실행은 저장소 루트에서 하세요** — `uoi_common.py`가 자기 위치 기준으로
`data/` 경로를 만들기 때문에, 이 폴더 안에서 실행하면 `main_code/data/`가
새로 생깁니다.

## 구성

| 단계 | 파일 | 메소드 |
|------|------|--------|
| 공통 | `uoi_common.py` | 공유 경로, Gini 계수 |
| 2 | `02_compute_uoi.py` | 4차원 UOI (connectivity / efficiency / accessibility / equity) |
| 2 | `02_compute_uoi_spec.py` | **6개 지표**: link-node ratio, connected-node ratio, intersection density, median block length, walking circuity(OD 샘플), pedshed reach(H3 격자 + 400 m ego-graph) |
| 3 | `03_stratified_sample.py` | 형태 유형 층화(KMeans k=4) + Pareto frontier 표본 |
| 3 | `viz_top1000.py` | 지표 → 전국 백분위 → composite UOI_score, top-1000 선정 |
| 4 | `04_sampler.py` | RJ-MCMC 코어: 5종 이동 + Hastings 보정, 평면성·연결성 제약, parallel tempering, split R-hat, hypervolume shortfall |
| 5 | `05_mcmc_spec.py` | 6지표 tanh-포화 에너지로 top-1000 반사실 탐색 → distance-to-frontier |
| 6 | `06_synthesize.py` | 벤치마크(top-1000 중앙값) 지향 reach-or-better 에너지로 가상 네트워크 합성 |
| 7 | `07_gnn_surrogate.py` | GraphSAGE 서로게이트 (G → dtf 회귀), 전국 ~84k tract 예측 |
| 8 | `08c_correlate.py` | UOI × 사회경제 아웃컴 Spearman 상관 |
| 8 | `08d_regression.py` | 밀도·소득·인종 + state FE 통제 표준화 OLS |
| 8 | `08e_ped_safety.py` | 보행자 안전 분해: 노출 검정 / 구성지표 효과 / FARS 메커니즘 프로파일 |
