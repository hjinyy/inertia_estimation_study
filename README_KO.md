# WMU 기반 다중 시간척도 + Multitaper 관성 추정 검증

## 이번 검증의 질문

SG가 알려진 시스템 기준 관성 2.8초를 가지며, ESS-VSG 및 PV-VSG의 내부 지연과 필터 계수는 추정기에서 알 수 없다고 가정했습니다. **같은 WMU 전압·전류 파형**에서 계산한 주파수·출력전력을 입력으로 사용하여, (i) 대표적인 시스템 식별 방식 ARX와 (ii) DPSS Multitaper·다중 시간척도·물리적 비음수 응답 제약을 결합한 후보 방법을 비교합니다. **정답(True)은 데이터 생성기에만 존재하며 추정기에 입력하지 않습니다.**

> 중요: 이번 코드의 ARX는 일반적인 ARX 시스템 식별 기준 모델입니다. Tuttelberg (2018)이나 이후 ARMAX 논문의 전체 추정 절차를 재구현한 것은 아닙니다. 결과 차이를 '기존 모든 관성 추정 논문보다 정확하다'고 주장할 수 없습니다.

## 실행

Python 3.10+에서:

```bash
pip install numpy scipy pandas matplotlib
python run_wmu_multitaper_vs_arx.py
```

별도의 Ambient 무작위 실험:

```bash
WMU_SEED=20261020 python run_wmu_multitaper_vs_arx.py
WMU_SEED=20261021 python run_wmu_multitaper_vs_arx.py
WMU_SEED=20261022 python run_wmu_multitaper_vs_arx.py
WMU_SEED=20261023 python run_wmu_multitaper_vs_arx.py
WMU_SEED=20261024 python run_wmu_multitaper_vs_arx.py
python aggregate_repeats.py
```

Windows PowerShell에서는 다음처럼 환경변수를 변경합니다.

```powershell
$env:WMU_SEED = "20261020"
python run_wmu_multitaper_vs_arx.py
```

동일한 난수 시드에서 동일한 결과를 생성합니다. `aggregate_repeats.py`는 위 다섯 시드 결과가 존재해야 실행됩니다. ZIP에는 생성된 5회 결과 요약이 포함되어 있습니다.

## 신호 생성 및 실험 환경

| 항목 | 설정 |
|---|---|
| 기준 주파수 | 60 Hz |
| 시스템 기준용량 | 500 MVA |
| 알려진 SG 관성 | 2.8 s (시스템 기준) |
| 전체 구간 | 4 x 120 s = 480 s |
| 파형 | 15,360 Hz, 3상 전압·전류 및 약한 5차 고조파 |
| 추정기 입력 | 파형으로부터 유도한 주파수·ESS/PV 출력전력, 240 Hz |
| 분석 Window | Stage별 15초~종료 8초 전 (97초/Stage) |
| Ambient 변동 | 랜덤부하 0.13~9.5 Hz, **고주파 에너지가 충분한 인위적 조건** |
| 관성 정답 | `H_obs`: 상세 합성 모델의 관성형 경로를 0.4~7 Hz에서 요약한 기준값 |
| Stage 1~4 설정 `H_set` | GFM 기준 0.82/1.22/0.68/0.00 s |

실제 **IEEE 39-bus EMT 및 상용 VSG 제어기 시뮬레이션이 아닙니다**. SG-ESS-VSG-PV-VSG의 간소화된 폐루프 연계모델이며, 실제 조속기·LCL·전류 제한기·통신지연·병입정보 변화는 미구현입니다.

## 비교 대상은 세 가지뿐입니다

1. **True** — 합성 모델 내부 관성형 전력 경로의 대역 평균으로 정의한 `H_obs`. 평가에만 사용합니다.
2. **Conventional ARX baseline** — 동일한 WMU 240Hz 파생 주파수·단자전력으로 선형 ARX 모델을 동적 식별한 뒤, 전력–주파수 전달함수의 허수부에서 겉보기 관성 산출. ARX 모델 차수는 시간순 검증 데이터로 선택합니다.
3. **Proposed WMU DPSS Multitaper** — 동일한 WMU 240Hz 데이터를 DPSS 다중테이퍼로 스펙트럼 추정 후, Droop형 `1/(1+s*tau)` 및 관성형 `s/(1+s*tau)` 성분을 여러 시간상수로 펼치고 비음수 제약 및 iterative reweighted ridge로 적합합니다. 지연·패널티는 독립 시간블록에서 검증한 오차로 선택합니다. 적합 결과에서 관성형 경로만 분리한 `H_obs`를 산출합니다.

**두 방법에 정확히 같은 WMU 특징량을 입력**하므로 결과 차이가 센서/PMU 필터 길이 차이가 아닌 *추정 방법 차이*를 보다 직접 반영합니다. 이번 비교만으로 WMU 자체가 표준 PMU보다 반드시 더 좋다는 주장은 불가능합니다. 해당 주장은 추후 같은 알고리즘으로 WMU와 적합한 PMU 추출기 데이터를 비교해야 가능합니다.

## 추정 신뢰도

- 파형 주파수응답: DPSS `NW=3`, 테이퍼 4개, 독립 8.53초 블록 11개를 평균.
- 모델 선택: 독립 시간블록 2/3 훈련, 1/3 검증(관측대역 0.25~9Hz).
- 모델 파라미터: 관성 및 Droop 응답 계수의 비음수 제약, 여러 시정수 dictionary, 후보 지연 0~45ms.
- 불확실성: 시간블록 재표본추출 18회로 얻은 **조건부** Bootstrap 변동, 여기에 시간블록 검증 잔차를 이용한 **휴리스틱 모델불일치 폭**을 추가합니다.
- Stage 4에서 추정값이 0과 양립하면 `ZERO_COMPATIBLE`로 표시해 확실한 양의 관성이라고 주장하지 않습니다.

**주의:** Bootstrap 결과만의 95% 구간은 모델 편향을 놓쳤습니다(5시드 × 4 Stage 중 참값 포함 0/20). 그래서 Figure에 표시한 음영은 신뢰수준이 교정된 95% 신뢰구간이 **아닌**, 모델 불일치를 고려한 *휴리스틱 민감도 대역*입니다. 이 폭을 이용한 PASS 상태를 실계통 신뢰성 보증으로 해석하면 안 됩니다. 5개 시드에서 해당 대역은 20/20 참값을 포함하지만, 교정에 충분한 표본이 아닙니다.

## 다섯 개 부하변동 시드 × 4 Stage = 20개 평가 결과

| 방법 | RMSE (s) | MAE (s) | Bias (s) |
|---|---:|---:|---:|
| 일반 ARX | 0.04523 | 0.04471 | -0.04471 |
| 제안 WMU Multi-timescale+DPSS | 0.00585 | 0.00491 | +0.00491 |

동일한 `H_obs` 정답에 대해 RMSE가 **87.1% 감소**했습니다. 이 결과는 이 합성모델과 설정 대역에서만 성립합니다. Stage 4(실제 GFM 관성=0)에서 일반 ARX는 평균 -0.0510 s의 가짜 관성형 응답을 보였고, 제안 방법의 추정값은 평균 +0.00136 s였습니다.

`H_set` (제어기 설정값)과 `H_obs` (외부 관측 정의)는 달라서 직접 비교하면 안 됩니다. 코드의 주요 그래프는 모두 **True H_obs**와 비교합니다.

## Figure 설명

- `FigA_five_seed_true_baseline_proposed.png`: 네 Stage의 평균 True·ARX·제안값과 하단 True 대비 평균오차. 오차막대는 5개 시드 간 표준편차이며, 푸른 음영은 모델불일치 휴리스틱 범위입니다.
- `FigB_RMSE_reduction.png`: 전체 20개 평가 RMSE 두 방법 비교 및 오차감소율.
- `FigC_ESS_PV_true_baseline_proposed.png`: ESS-VSG 및 PV-VSG 기여량을 True와 함께 각각 비교.
- `FigD_system_total_true_baseline_proposed.png`: 알려진 SG 2.8s를 더한 계통 전체 관성 비교. 합성모델에서 같은 대역 정의라는 전제가 있습니다.
- `results_seed_20261020/Fig3_true_frequency_resolved_response.png`: 한 시드의 Stage 1/4에서 주파수별 True 및 제안·ARX 결과 비교.

## 파일

- `run_wmu_multitaper_vs_arx.py`: 독립 실행형 재현 코드
- `aggregate_repeats.py`: 5시드 집계 및 최종 Fig A~D 재생성
- `results_5seed_summary/`: 최종 Figure 4개, 집계 CSV, 메타데이터
- `results_seed_*/`: 시드별 오차표, 장치별 지연·Droop 진단 CSV(기본 시드는 모든 Figure, 다른 시드는 요약 표만 포함)

## 논문 작성 시 제한사항

1. 이번 연구는 파형 계측에 기반하지만, 실제 추정기는 240Hz로 정리한 `P,f`만 입력받습니다. **고조파·Sub-cycle 특징 자체가 식별성 향상을 주었다는 증명은 아닙니다.**
2. 제안법은 '완전 무모델 블랙박스'가 아닙니다. 가상관성/ Droop의 외부 일반 응답이 각각 주파수 미분 및 편차에 관련된다는 물리적 가정이 있습니다.
3. 120초 단위 고정구간 Batch 추정으로, 정해진 시간마다 새 결과를 내는 구조입니다. **아직 Sliding/Moving Window 기반의 엄격한 Continuous Online 검증이 아닙니다.**
4. 사용한 Ambient 신호는 9.5Hz까지 충분한 에너지를 갖도록 인위적으로 생성했습니다. 실제 계통 주변변동은 훨씬 느리거나 관성 파라미터를 유일하게 구분하지 못할 수 있습니다.
5. ARX는 표준적인 식별 기준선이지만 본문 선행연구에 등장하는 ARMAX 알고리즘 전체를 복제하지 않았습니다. 선행연구 대비 논문 성과 주장을 위해서는 해당 방법의 독립 구현이나 공개 코드와 함께 추가 비교가 필요합니다.
6. 추정 성공 조건을 확정하려면 더 많은 잡음 시드, 모델 구조 변경, 통신지연·측정 누락·추가 GFM·제약 포화 등을 실험해야 합니다.

## 참고 문헌

- Tuttelberg et al. (2018), *Estimation of Power System Inertia From Ambient Wide Area Measurements*, IEEE Transactions on Power Systems, https://doi.org/10.1109/TPWRS.2018.2843381
- Ramirez-Gonzalez et al. (2026), *Continuous real-time inertia estimation in power systems using ARMAX, ARX, and OE models*, Energy Reports, https://www.sciencedirect.com/science/article/pii/S2352484726002994

