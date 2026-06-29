# UOI Street-Network Pipeline — 코드 가이드 (인수인계용)

미국 전체 census tract(~84,400개)의 **UOI(Urban Optionality Index)** 를 계산하고,
MCMC로 최적 네트워크를 탐색하고, GNN으로 전국을 예측하는 연구 파이프라인.

실행 환경: conda env **`street`** (python 3.12, osmnx 2.1, geopandas 1.1, pyrosm 0.8, torch 2.5+cu121, PyG 2.8)
```
source ~/anaconda3/etc/profile.d/conda.sh && conda activate street
```
파일명 앞 숫자 = 실행 순서(stage). `data/`, `.claude/` 는 gitignore.

---

## 파이프라인 흐름
```
Stage 1  네트워크 추출      .pbf → tract별 보행 그래프(.graphml)
Stage 2  UOI 6지표 계산     그래프 → uoi_spec_metrics.parquet
Stage 3  층화 표집          타이폴로지 군집 + Pareto + 분석 샘플
Stage 4/5 MCMC 최적망 탐색  tract별 최적 대안망 + dtf(최적까지 거리)
Stage 5b 가상망 합성        도시 무관 최적 가상 네트워크
Stage 6/7 GNN 대리모델      dtf 학습 → 전국 84k 예측
viz_*    각 단계 시각화
```

---

## 공통 / 유틸
| 파일 | 설명 |
|------|------|
| **uoi_common.py** | 공유 경로·헬퍼(디렉터리 상수 등). 모든 스크립트가 import. |

## Stage 1 — 네트워크 추출 (변형이 많은 이유 = 메모리/속도 문제 해결사)
모든 추출기는 **resumable**(이미 만든 .graphml은 건너뜀). 변형이 6개인 건 큰 주(CA/TX 등)에서
pyrosm 파싱이 OOM/초저속이라 그걸 우회하려고 진화한 결과.

| 파일 | 설명 | 용도 |
|------|------|------|
| **01_extract_networks.py** | **(구버전)** Overpass API로 county 단위 다운로드 후 tract로 분할. 전국엔 너무 불안정(타임아웃). | 소규모/검증용. 사실상 폐기 |
| **01_extract_networks_pbf.py** | **주력 추출기.** Geofabrik state `.pbf`를 county별로 bbox 파싱 → 그래프 빌드+simplify → tract별 분할. county 단위라 메모리 안정(~3–7GB). | 대부분의 주 |
| **01b_extract_mega.py** | state `.pbf`를 **한 번만** 파싱해 메모리에서 tract별 슬라이스. county 반복 파싱을 없앰. | 중대형 주 (IL: 10.3M노드/47GB OK) |
| **01c_extract_batched.py** | county들을 RAM에 맞는 **배치**로 묶어 각 배치를 별도 subprocess에서 처리(OOM 격리). | CA/TX 시도 (polygon bbox라 느림 → 아래로 대체) |
| **01d_extract_strips.py** | 주를 **경도 띠(직사각형 bbox)** 로 잘라 파싱. polygon bbox의 병목 회피. | CA/TX 대안 |
| **01e_extract_osmium.py** | `osmium extract`(C++)로 .pbf를 작은 **타일**로 쪼갠 뒤 각 타일을 bbox 없이(=빠른 경로) 파싱. **CA/TX 최종 해법.** | 메가주 권장 |

> 핵심 교훈: pyrosm은 bbox(polygon/rect 무관)를 주면 element별 필터링으로 **병리적으로 느려짐**.
> bbox 없는 전체 파싱은 빠르지만 메가주는 OOM. → osmium으로 미리 잘라 빠른 경로를 쓰는 01e가 정답.

## Stage 2 — UOI 지표 계산
| 파일 | 설명 |
|------|------|
| **02_compute_uoi.py** | **(구버전)** 4차원 UOI(connectivity/efficiency/accessibility/equity). design-doc과 불일치로 대체됨. |
| **02_compute_uoi_spec.py** | **현행.** design-doc "UOI Index" 표의 **6지표**: ①link-node ratio ②connected-node ratio ③intersection density ④median block length ⑤walking circuity ⑥pedshed reach. 각 권고범위 충족 플래그(`*_ok`) 포함. **단일 진실 소스.** |
| **02b_compute_uoi_parallel.py** | 02_spec의 지표 코드를 그대로 재사용하며 multiprocessing으로 병렬화(~선형 스케일). resumable, GEOID 해시로 RNG 시드 고정(재현성). 전국 계산용. |

## Stage 3 — 층화 표집
| 파일 | 설명 |
|------|------|
| **03_stratified_sample.py** | 형태 특징으로 KMeans(k=4) 군집 → gridded/cul-de-sac/organic/hybrid 라벨, UOI Pareto frontier 표시, 층화 샘플 추출(MCMC 심층분석 대상). |

## Stage 4 / 5 — RJ-MCMC 최적 네트워크 탐색
| 파일 | 설명 |
|------|------|
| **04_sampler.py** | **MCMC 엔진.** Reversible-Jump MCMC + parallel tempering. tract 폴리곤 안에서 물리적으로 타당한 도로망 공간을 탐색해 고-UOI 대안망 발견. 이동(move): 노드이동/엣지추가·삭제/엣지분할/degree-2 병합. 4차원 UOI 기준. |
| **05_mcmc_spec.py** | **현행 MCMC.** 04의 RJ-MCMC 기계장치(이동·Hastings비·tempering)를 **그대로 재사용**하고 평가자만 **6지표 spec + 권고범위**로 교체. tract별 `dtf`(distance-to-frontier, 0=이미 최적) 산출. top-1000 + 전국샘플 800에 실행 → summary.json(1800). |

## Stage 5b — 가상망 합성
| 파일 | 설명 |
|------|------|
| **06_synthesize.py** | 실제 tract 없이 **빈 정사각 도메인에서 도시 무관 최적 네트워크를 새로 생성**. 04 move + 05 평가자 재사용, 에너지는 top-1000 중앙값 목표 대비 절대 "reach-or-better". 시드 3종(grid/organic/hybrid). grid가 최고점 = 격자형이 UOI 최적임을 검증. |

## Stage 6 / 7 — GNN 대리모델
| 파일 | 설명 |
|------|------|
| **07_gnn_surrogate.py** | **GNN.** `train`: MCMC 라벨(summary.json의 dtf)로 GraphSAGE×3+global pool+6지표 → dtf 회귀(수 분/tract → 수 ms/tract). `predict`: 전국 84k .graphml 예측 → gnn_dtf_predictions.parquet. OOD 플래그+클램프 내장. 노드특징은 위상(topology)만이라 추론 시 지표 불필요. |

## 시각화
| 파일 | 설명 |
|------|------|
| **viz_uoi.py** | Stage2(구 4차원) 상관 히트맵 + SF 코로플레스 + 표 → results/figures, tables |
| **viz_state.py** | 한 주의 추출 그래프로 UOI 계산 + 히트맵/지도 → results/state_<FIPS>/ (주별 쇼케이스) |
| **viz_network_uoi.py** | 몇몇 county의 실제 도로망 + UOI 코로플레스 나란히 → 숫자의 현실 의미 |
| **viz_top1000.py** | 6지표를 전국 백분위로 변환→평균=UOI 합성점수, 전체 순위→top-1000 데이터+그림 |
| **viz_sampler.py** | (Stage4) 실제 vs 최적 대안망 + posterior UOI 구름 + Pareto frontier |
| **viz_mcmc_spec.py** | (Stage5) dtf 표/분포, 지표 shift(권고범위 표시), 실제 vs 최적망 갤러리 → results/mcmc_spec/ |

## 산출물 묶음
| 파일 | 설명 |
|------|------|
| **build_export.py** | 모든 단계의 그림+데이터를 `results/export_<날짜>/`(6단계 폴더+README+SUMMARY_STATS.json)로 통합, 전국 헤드라인 통계 계산. |

---

## Shell 드라이버 (detached 실행, resumable)
대부분 `setsid nohup`으로 백그라운드 실행, 로그는 `data/outputs/`(또는 state_logs/).

| 파일 | 설명 |
|------|------|
| **run_all_states_pbf.sh** | 전국 pbf 추출(stage1 전체 주 → stage2). 초기 주력 러너. |
| **run_parallel_pbf.sh** | `xargs -P6`로 6개 주 동시 추출(큰 주는 간격 배치해 메모리 안전). **권장 병렬 러너.** |
| **run_rest_states_pbf.sh** | 큰 주 제외하고 나머지 주만 순차 추출(구버전). |
| **run_megastates_batched.sh** / **run_ca_after_tx.sh** | CA/TX 메가주 배치 추출 + TX 후 CA 자동시작. |
| **run_strips.sh** / **run_osmium_extract.sh** | 01d(띠) / 01e(osmium 타일) 추출 구동. |
| **run_finish_remaining.sh** / **run_retry_failed.sh** | 미완/실패 tract 백필·재시도. |
| **run_mcmc_spec.sh** | Stage5 MCMC 구동 `[TOP ITERS WEIGHTS REPLICAS TEMPS PROCS]`. |
| **run_all_states.sh** | (구) Overpass 버전. 폐기. |

---

## 빠른 재현 순서 (요약)
```bash
conda activate street
# 1) 추출(대부분 주)             ./run_parallel_pbf.sh
#    메가주(CA/TX)               ./run_osmium_extract.sh   (01e)
# 2) 6지표 계산(병렬)            python 02b_compute_uoi_parallel.py --workers 24
# 3) top-1000 / 샘플             python viz_top1000.py ; python 03_stratified_sample.py
# 4) MCMC 최적망(top-1000)       ./run_mcmc_spec.sh 1000 4000 2 2 4 24
# 5) 시각화                       python viz_mcmc_spec.py
# 6) GNN 학습 → 전국 예측         python 07_gnn_surrogate.py train
#                                python 07_gnn_surrogate.py predict --procs 24
# 7) 산출물 묶음                  python build_export.py
```
```
```
