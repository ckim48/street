# main_code — 핵심 파이프라인 코드 모음

UOI street-network 파이프라인에서 실제 결과 산출에 사용된 주요 스크립트만
저장소 루트에서 복사해 정리한 폴더입니다 (레퍼런스/아카이브용 스냅샷).
**실행은 저장소 루트에서 하세요** — `uoi_common.py`가 자기 위치 기준으로
`data/` 경로를 만들기 때문에, 이 폴더 안에서 실행하면 `main_code/data/`가
새로 생깁니다. 방법론과 수식 정리는 `docs/METHODS_UOI.docx` 참조.

## 파이프라인 순서

| 순서 | 파일 | 역할 |
|------|------|------|
| 공통 | `uoi_common.py` | 공유 경로, TIGER tract 로더, Gini 계수 |
| 1 | `01_extract_networks.py` | Overpass 기반 추출 (SF 파일럿, county 단위 쿼리) |
| 1 | `01_extract_networks_pbf.py` | **전국 스케일 기본 경로** — Geofabrik state .pbf + pyrosm, county 단위 파싱 → tract별 GraphML |
| 1 | `run_all_states_pbf.sh` | 전 주(state) .pbf 추출 드라이버 |
| 2 | `02_compute_uoi.py` | 4차원 UOI (connectivity/efficiency/accessibility/equity) — 파일럿·층화표집용 |
| 2 | `02_compute_uoi_spec.py` | **설계문서 6개 지표** (LNR, CNR, intersection density, block length, circuity, pedshed) |
| 3 | `03_stratified_sample.py` | 형태 유형(KMeans k=4) 층화 + Pareto frontier 표본 |
| 3 | `viz_top1000.py` | 지표 → 전국 백분위 → composite UOI_score, top-1000 선정 |
| 4 | `04_sampler.py` | RJ-MCMC 코어 (이동/Hastings/평면성 제약/parallel tempering), 4차원 에너지 |
| 5 | `05_mcmc_spec.py` | 6지표 spec 에너지로 top-1000 반사실 탐색 → distance-to-frontier |
| 5 | `run_mcmc_spec.sh` | MCMC 실행 드라이버 |
| 6 | `06_synthesize.py` | 벤치마크(top-1000 중앙값) 지향 가상 네트워크 합성 |
| 7 | `07_gnn_surrogate.py` | GraphSAGE 서로게이트로 dtf를 전국 ~84k tract에 확장 |
| 8a | `08a_fetch_external.sh` | 외부 데이터 다운로드 (Opportunity Atlas, Eviction Lab, FARS, LODES) |
| 8b | `08b_build_tract_panel.py` | GEOID 키 tract 패널 구축 |
| 8b | `08b_acs.py` | ACS 인구·소득·인종·학력 결합 |
| 8c | `08c_correlate.py` | UOI × 아웃컴 상관 (Spearman, hexbin) |
| 8d | `08d_regression.py` | 밀도·소득·인종 + state FE 통제 조정 회귀 |
| 8e | `08e_ped_safety.py` | 보행자 안전 심층 분석 (노출/구성지표/FARS 메커니즘) |

## 제외된 것

- `01b`–`01e` 추출 변형(mega/batched/strips/osmium): 대형 주 처리용 임시 우회 버전
- `viz_*.py` 나머지: 그림 생성 전용 (루트에 유지)
- `run_*` 나머지: 상태별 재시도/부분 실행 드라이버
