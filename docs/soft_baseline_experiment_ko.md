# Continuous Soft Allocation Baseline 실험 설명

## 목적

- 기존 논문 방식은 binary user-RB assignment를 Hungarian matching으로 결정한다.
- 본 실험은 동일한 wireless FL 환경에서 hard assignment를 continuous soft allocation으로 완화한 두 baseline을 추가해 비교한다.
- 추가 baseline은 `Continuous-LP(HiGHS)`와 `Soft-Entropy-KKT(tau=1)`이다.
- 결과 분석은 `outputs/experiment_3`의 JSON 및 log 결과를 기준으로 작성했다.

## 공통 설정

- 공통 목적: feasible user-RB edge 중 FL 수렴률 surrogate에 유리한 조합을 선택한다.
- edge cost:

```text
c_{i,n} = K_i (q_{i,n} - 1)
```

- `K_i`: user `i`의 local data 수
- `q_{i,n}`: user `i`가 RB `n`을 사용할 때의 packet error rate
- feasible하지 않은 edge는 allocation에서 제외한다.
- Fig. 7-9는 MNIST `test_samples=10000`, seed `0..9`, 130 communication rounds 조건이다.
- Fig. 4/7/8/9에 대해서만 soft baseline을 추가했다. Fig. 10은 논문 예시 이미지 비교이므로 soft baseline 비교 대상이 아니다.

## 이론적 시간 복잡도

### 기호 정의

- `U`: 사용자 수
- `R`: RB 수
- `N = max(U, R)`
- `B`: 각 user-RB edge의 송신 전력 탐색 반복 수
- `T`: FL communication round 수
- `E`: local epoch 수
- `K_i`: user `i`의 local sample 수
- `C_model`: sample 1개에 대한 forward/backward 학습 비용
- `I_sinkhorn`: Soft-Entropy-KKT의 Sinkhorn/KKT scaling 반복 수

### 자원 할당 단계 복잡도

모든 방법은 먼저 feasible edge를 만들기 위해 `U x R` link matrix를 계산한다. 각 edge에서 optimal transmit power, rate, packet error, delay, energy를 계산하므로 공통 비용은 대략 `O(U R B)`이다.

| Method | Assignment solver 복잡도 | 자원 할당 전체 복잡도 |
| --- | ---: | ---: |
| Proposed Hungarian | `O(N^3)` | `O(U R B + N^3)` |
| Baseline a | `O(N^3)` | `O(U R B + N^3)` |
| Baseline b | `O(U + R)` | `O(U R B + U + R)` |
| Baseline c | `O(N^3)` | `O(U R B + N^3)` |
| Continuous-LP(HiGHS) | solver-dependent | `O(U R B + LP(U R, U+R))` |
| Soft-Entropy-KKT(tau=1) | `O(I_sinkhorn (U+R)^2)` | `O(U R B + I_sinkhorn (U+R)^2)` |

### Baseline별 해석

- Proposed Hungarian은 FL-aware cost `K_i(q_{i,n}-1)`를 만든 뒤 Hungarian/Munkres matching을 수행한다. matching 자체는 cubic complexity `O(N^3)`이다.
- Baseline a는 FL-aware user set을 고르는 과정에서 Hungarian matching을 사용하고 RB를 randomize하므로 asymptotic assignment 비용은 Proposed와 동일하다.
- Baseline b는 selected user와 RB를 random으로 고르므로 solver 비용은 `O(U+R)` 수준이다. 단, 현재 구현에서는 공정 비교를 위해 wireless matrix를 여전히 계산하므로 전체 비용에는 `O(U R B)`가 포함된다.
- Baseline c는 wireless-only cost로 Hungarian/Munkres를 사용하므로 `O(N^3)`이다.
- Continuous-LP(HiGHS)는 변수 수가 `U R`, row/column 제약 수가 `U+R`인 LP를 푼다. assignment LP는 구조적으로 special case이지만, 구현은 generic HiGHS LP solver를 사용하므로 단순히 Hungarian보다 낮은 차수라고 말하기 어렵다.
- Soft-Entropy-KKT는 dummy row/column을 포함한 대략 `(U+R) x (U+R)` matrix에 Sinkhorn scaling을 반복한다. 반복 수 `I_sinkhorn`이 작고 안정적이면 assignment 단계만 보면 cubic Hungarian보다 가볍게 동작할 수 있다.

### FL 학습까지 포함한 복잡도

자원 할당 이후에는 local training과 aggregation 비용이 지배적일 수 있다. hard assignment 계열의 학습 비용은 대략 다음과 같다.

```text
O(T E sum_{i in selected} K_i C_model)
```

soft baseline은 fractional mass가 여러 user에 퍼질 수 있으므로 다음처럼 볼 수 있다.

```text
O(T E sum_{i in soft-selected} K_i C_model)
```

따라서 assignment solver만 보면 `Soft-Entropy-KKT`가 Hungarian보다 가벼울 수 있지만, soft-selected user 수가 많아지면 전체 FL wall-clock 시간은 오히려 증가할 수 있다. 반대로 `Continuous-LP`는 solver가 generic LP이므로 이론적 solver 비용은 명확히 낮다고 보기 어렵지만, assignment polytope 특성상 성능은 Hungarian과 매우 가까운 해를 낸다.

## 물리적 실행 시간 결과

### 측정 방식

- `experiment_3`부터 각 method runner 호출의 wall-clock seconds를 `method_wall_time_seconds`에 저장했다.
- 해당 필드는 각 figure의 JSON, `run_all.log`, `figure_N/run.log`에 기록된다.
- 측정 범위는 method runner 내부의 wireless 계산, assignment, FL 학습, evaluation이다.
- plotting, JSON 저장, log 저장 시간은 각 method timing에 포함하지 않는다.
- 따라서 이 값은 순수 assignment solver 시간만이 아니라 실제 실험에서 체감되는 end-to-end method 실행 시간이다.

### Figure별 평균 실행 시간

| Figure | Method | Runs | Mean sec/run | Total sec | Proposed 대비 |
| --- | --- | ---: | ---: | ---: | ---: |
| Fig. 4 | Proposed | 50 | 2.446 | 122.289 | 1.000 |
| Fig. 4 | Baseline a | 50 | 2.435 | 121.750 | 0.996 |
| Fig. 4 | Baseline b | 50 | 2.426 | 121.296 | 0.992 |
| Fig. 4 | Continuous-LP(HiGHS) | 50 | 2.555 | 127.751 | 1.045 |
| Fig. 4 | Soft-Entropy-KKT(tau=1) | 50 | 3.103 | 155.162 | 1.269 |
| Fig. 7 | Proposed | 10 | 11.923 | 119.231 | 1.000 |
| Fig. 7 | Baseline a | 10 | 11.856 | 118.561 | 0.994 |
| Fig. 7 | Baseline b | 10 | 13.118 | 131.178 | 1.100 |
| Fig. 7 | Baseline c | 10 | 12.871 | 128.710 | 1.080 |
| Fig. 7 | Continuous-LP(HiGHS) | 10 | 11.893 | 118.932 | 0.997 |
| Fig. 7 | Soft-Entropy-KKT(tau=1) | 10 | 12.694 | 126.935 | 1.065 |
| Fig. 8 | Proposed | 60 | 11.041 | 662.438 | 1.000 |
| Fig. 8 | Baseline a | 60 | 10.988 | 659.284 | 0.995 |
| Fig. 8 | Baseline b | 60 | 11.861 | 711.671 | 1.074 |
| Fig. 8 | Baseline c | 60 | 10.852 | 651.130 | 0.983 |
| Fig. 8 | Continuous-LP(HiGHS) | 60 | 11.083 | 664.964 | 1.004 |
| Fig. 8 | Soft-Entropy-KKT(tau=1) | 60 | 11.631 | 697.850 | 1.053 |
| Fig. 9 | Proposed | 40 | 11.028 | 441.129 | 1.000 |
| Fig. 9 | Baseline a | 40 | 10.964 | 438.540 | 0.994 |
| Fig. 9 | Baseline b | 40 | 11.386 | 455.420 | 1.032 |
| Fig. 9 | Baseline c | 40 | 10.696 | 427.850 | 0.970 |
| Fig. 9 | Continuous-LP(HiGHS) | 40 | 11.002 | 440.097 | 0.998 |
| Fig. 9 | Soft-Entropy-KKT(tau=1) | 40 | 11.796 | 471.855 | 1.070 |

### Soft baseline 시간 분석

- `Continuous-LP(HiGHS)`는 Fig. 7과 Fig. 9에서는 Proposed와 거의 같은 시간이 걸렸다.
  - Fig. 7: Proposed 대비 `0.997x`
  - Fig. 8: Proposed 대비 `1.004x`
  - Fig. 9: Proposed 대비 `0.998x`
- Fig. 4에서는 `Continuous-LP`가 Proposed보다 `1.045x` 느렸다. 작은 regression 실험에서는 solver overhead가 상대적으로 더 크게 보인다.
- `Soft-Entropy-KKT(tau=1)`는 모든 figure에서 Proposed보다 느렸다.
  - Fig. 4: `1.269x`
  - Fig. 7: `1.065x`
  - Fig. 8: `1.053x`
  - Fig. 9: `1.070x`
- Soft-Entropy-KKT의 이론적 assignment 비용은 반복 scaling 기반이지만, 실제 end-to-end 시간에서는 soft-selected user 처리와 반복 scaling overhead가 더해져 Proposed보다 느린 결과가 나왔다.
- 전체적으로 물리적 시간 기준에서는 `Continuous-LP`가 Proposed와 가장 가까운 비용을 보이고, `Soft-Entropy-KKT`는 성능 차이가 작음에도 실행 시간이 더 높다.

### Sweep 축별 soft baseline 실행 시간

Fig. 4 sample count별 평균 seconds/run:

| Samples/user | Proposed | Continuous-LP | Soft-Entropy-KKT |
| ---: | ---: | ---: | ---: |
| 10 | 2.115 | 2.220 | 2.726 |
| 20 | 2.279 | 2.419 | 2.928 |
| 30 | 2.441 | 2.551 | 3.116 |
| 40 | 2.623 | 2.712 | 3.284 |
| 50 | 2.771 | 2.874 | 3.462 |

Fig. 8 user count별 평균 seconds/run:

| Users | Proposed | Continuous-LP | Soft-Entropy-KKT |
| ---: | ---: | ---: | ---: |
| 3 | 9.365 | 9.371 | 9.391 |
| 6 | 10.453 | 10.486 | 10.766 |
| 9 | 10.833 | 10.917 | 11.280 |
| 12 | 11.434 | 11.438 | 12.107 |
| 15 | 11.918 | 12.020 | 12.829 |
| 18 | 12.241 | 12.266 | 13.411 |

Fig. 9 RB count별 평균 seconds/run:

| RBs | Proposed | Continuous-LP | Soft-Entropy-KKT |
| ---: | ---: | ---: | ---: |
| 3 | 9.896 | 9.906 | 10.783 |
| 6 | 10.807 | 10.814 | 11.440 |
| 9 | 11.394 | 11.326 | 12.176 |
| 12 | 12.015 | 11.963 | 12.786 |

## Baseline 1: Continuous-LP(HiGHS)

### 핵심 아이디어

- binary variable `x_{i,n} in {0,1}`을 continuous variable `0 <= x_{i,n} <= 1`로 relaxation한다.
- power optimization으로 계산된 `q_{i,n}`을 고정한 뒤, assignment만 linear programming으로 푼다.
- LP는 다음 형태다.

```text
minimize    sum_i sum_n c_{i,n} x_{i,n}
subject to  sum_n x_{i,n} <= 1       for each user i
            sum_i x_{i,n} <= 1       for each RB n
            x_{i,n} = 0              if edge (i,n) is infeasible
            0 <= x_{i,n} <= 1
```

### 구현 방식

- `scipy.optimize.linprog(method="highs")`를 사용한다.
- LP relaxation은 assignment polytope 구조를 가지므로 선형 목적함수에서는 최적해가 대체로 hard matching과 매우 가까운 extreme point로 나온다.
- FL aggregation에서는 hard rounding을 하지 않는다.
- user `i`의 expected success mass를 다음처럼 계산해 FedAvg 가중치에 반영한다.

```text
s_i = sum_n x_{i,n} (1 - q_{i,n})
```

### 결과 분석

- Fig. 4 final regression loss: `0.158832`
  - Proposed `0.158233`보다 `+0.000599` 높다.
  - Baseline b `0.158899`보다는 약간 낮다.
  - regression loss에서는 기존 proposed/Hungarian 대비 이점은 없고, 거의 동률 수준이다.

- Fig. 7 final MNIST accuracy: `0.85377`
  - Proposed `0.85374`보다 `+0.00003` 높다.
  - 130-round 평균 accuracy는 `0.76395`로 Proposed `0.76394`와 사실상 동일하다.
  - Baseline a/b/c보다 final accuracy가 각각 약 `+1.03`, `+0.53`, `+1.35` percentage point 높다.

- Fig. 8 user 수 sweep 평균 accuracy: `0.84321`
  - Proposed 평균 `0.84323`보다 `-0.00002` 낮다.
  - Baseline c 평균 `0.84220`보다는 높고, random 계열 baseline a/b보다 크게 안정적이다.
  - user 수가 18일 때는 `0.85506`으로 Proposed `0.85539`보다 약간 낮다.

- Fig. 9 RB 수 sweep 평균 accuracy: `0.85132`
  - Proposed 평균 `0.85123`보다 `+0.00009` 높다.
  - RB `[3, 6, 9, 12]` 전반에서 Proposed와 거의 같은 성능을 보인다.
  - Baseline c 평균 대비 약 `+0.23` percentage point, Baseline b 대비 약 `+3.25` percentage point 높다.

### 해석

- Continuous-LP는 목적함수와 제약이 Hungarian matching과 거의 같은 assignment 구조를 갖기 때문에 성능도 Proposed와 매우 유사하다.
- 차이는 주로 packet success를 stochastic draw로 처리하는지, expected success mass로 처리하는지에서 발생한다.
- 물리적 시간 기준으로는 Fig. 7-9에서 Proposed와 거의 같은 수준이며, Fig. 4에서만 solver overhead로 약간 느렸다.
- 따라서 이 baseline은 “continuous relaxation을 적용해도 기존 Hungarian matching 수준의 성능과 시간 비용을 유지할 수 있는가”를 확인하는 기준선으로 보는 것이 적절하다.

## Baseline 2: Soft-Entropy-KKT(tau=1)

### 핵심 아이디어

- assignment LP에 entropy regularization을 추가해 differentiable soft allocation을 만든다.
- hard matching 대신 여러 feasible edge에 fractional mass를 분산할 수 있다.
- `tau=1`은 entropy smoothing 강도를 의미한다.

### 구현 방식

- feasible edge에 대해 Gibbs kernel을 구성한다.

```text
G_{i,n} = exp(-c_{i,n} / tau)
```

- row/column slack을 dummy node로 확장한 뒤 Sinkhorn scaling으로 KKT 조건을 만족하는 soft assignment를 계산한다.
- `tau=1`에서는 cost scale이 `K_i(q_{i,n}-1)`로 비교적 크기 때문에 soft allocation이 완전 균등하게 퍼지기보다는 좋은 edge 주변에 집중된다.
- FL aggregation은 Continuous-LP와 동일하게 expected success mass `s_i`를 FedAvg 가중치로 사용한다.

### 결과 분석

- Fig. 4 final regression loss: `0.159018`
  - Proposed `0.158233`보다 `+0.000785` 높다.
  - Continuous-LP `0.158832`보다도 약간 높아, regression loss 기준으로는 soft entropy smoothing의 직접 이점은 작다.

- Fig. 7 final MNIST accuracy: `0.85384`
  - Proposed `0.85374`보다 `+0.00010` 높다.
  - 130-round 평균 accuracy는 `0.76397`로 Proposed보다 `+0.00003` 높다.
  - 차이는 0.01 percentage point 미만이므로 실질적으로 Proposed와 동률이다.

- Fig. 8 user 수 sweep 평균 accuracy: `0.84330`
  - Proposed 평균 `0.84323`보다 `+0.00007` 높다.
  - user 수 `[3, 6, 9, 12, 15, 18]`에서 Proposed와 거의 같은 곡선을 보인다.
  - 평균 기준으로는 추가 baseline 중 가장 높지만 차이가 매우 작다.

- Fig. 9 RB 수 sweep 평균 accuracy: `0.85140`
  - Proposed 평균 `0.85123`보다 `+0.00017` 높다.
  - RB 6, 9, 12에서는 Proposed보다 약간 높고, RB 3에서는 거의 동일하다.
  - Baseline c 평균보다 약 `+0.23` percentage point, Baseline b 평균보다 약 `+3.28` percentage point 높다.

### 해석

- Soft-Entropy-KKT는 continuous differentiable surrogate로서 가장 부드러운 baseline이다.
- MNIST accuracy 기준으로는 Proposed/Hungarian과 거의 같은 수준이며, Fig. 7-9 평균에서는 아주 근소하게 높다.
- 그러나 차이가 매우 작기 때문에 현재 결과만으로 soft entropy가 명확히 우월하다고 주장하기는 어렵다.
- 물리적 실행 시간은 Proposed 대비 `1.05x`에서 `1.27x` 수준으로 더 높았다. 성능 향상 폭이 매우 작다는 점을 고려하면, 현재 설정의 `tau=1`은 성능 대비 비용 효율이 높다고 보기는 어렵다.
- 의미 있는 결론을 내려면 seed별 분산, confidence interval, `tau` sweep이 추가로 필요하다.

## 주요 성능 결과 요약

### Fig. 4: regression loss

낮을수록 좋다. JSON에는 sample count sweep 중 final point가 저장되어 있다.

| Method | Final loss | Proposed 대비 |
| --- | ---: | ---: |
| Proposed | 0.158233 | 0.000000 |
| Baseline a | 0.157812 | -0.000422 |
| Baseline b | 0.158899 | +0.000666 |
| Continuous-LP(HiGHS) | 0.158832 | +0.000599 |
| Soft-Entropy-KKT(tau=1) | 0.159018 | +0.000785 |

### Fig. 7: round별 MNIST accuracy

| Method | Final accuracy | 130-round mean | Proposed 대비 final |
| --- | ---: | ---: | ---: |
| Proposed | 0.85374 | 0.76394 | 0.00000 |
| Baseline a | 0.84341 | 0.75554 | -0.01033 |
| Baseline b | 0.84848 | 0.76191 | -0.00526 |
| Baseline c | 0.84027 | 0.74479 | -0.01347 |
| Continuous-LP(HiGHS) | 0.85377 | 0.76395 | +0.00003 |
| Soft-Entropy-KKT(tau=1) | 0.85384 | 0.76397 | +0.00010 |

### Fig. 8: user 수별 MNIST accuracy

| Users | Proposed | Continuous-LP | Soft-Entropy-KKT | Baseline c | Baseline a | Baseline b |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.81042 | 0.81080 | 0.81073 | 0.81156 | 0.50717 | 0.50717 |
| 6 | 0.84219 | 0.84215 | 0.84240 | 0.83849 | 0.67508 | 0.74475 |
| 9 | 0.84683 | 0.84669 | 0.84675 | 0.84478 | 0.83234 | 0.83097 |
| 12 | 0.85081 | 0.85078 | 0.85073 | 0.84979 | 0.83682 | 0.84396 |
| 15 | 0.85374 | 0.85377 | 0.85384 | 0.85345 | 0.84341 | 0.84848 |
| 18 | 0.85539 | 0.85506 | 0.85534 | 0.85515 | 0.84971 | 0.84393 |
| Mean | 0.84323 | 0.84321 | 0.84330 | 0.84220 | 0.75742 | 0.76988 |

### Fig. 9: RB 수별 MNIST accuracy

| RBs | Proposed | Continuous-LP | Soft-Entropy-KKT | Baseline c | Baseline a | Baseline b |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.84733 | 0.84770 | 0.84765 | 0.84076 | 0.82921 | 0.78905 |
| 6 | 0.85085 | 0.85087 | 0.85097 | 0.84848 | 0.82299 | 0.81148 |
| 9 | 0.85300 | 0.85294 | 0.85314 | 0.85199 | 0.84107 | 0.82544 |
| 12 | 0.85374 | 0.85377 | 0.85384 | 0.85345 | 0.84341 | 0.84848 |
| Mean | 0.85123 | 0.85132 | 0.85140 | 0.84867 | 0.83417 | 0.81861 |

## 결론

- 두 soft baseline 모두 기존 Proposed/Hungarian과 거의 같은 성능을 보인다.
- `Continuous-LP(HiGHS)`는 assignment LP의 구조상 Hungarian과 유사한 해를 주며, end-to-end 실행 시간도 Fig. 7-9에서 Proposed와 거의 동일하다.
- `Soft-Entropy-KKT(tau=1)`는 differentiable surrogate라는 장점이 있고 Fig. 7-9 평균에서는 아주 근소하게 가장 높은 accuracy를 보이지만, 물리적 실행 시간은 Proposed보다 더 길다.
- Proposed 대비 성능 개선 폭은 대부분 `0.0001` 수준이므로, 현재 실험만으로 soft baseline의 성능 우위를 강하게 주장하기는 어렵다.
- 실용적 관점에서는 `Continuous-LP`가 “성능 유지 + 비용 유사” baseline으로 가장 안정적이다.
- 연구적 관점에서는 `Soft-Entropy-KKT`가 differentiable pipeline 확장 가능성을 보여주지만, `tau` sweep과 비용 대비 성능 분석이 추가로 필요하다.

## 결과 파일

- `outputs/experiment_3/run_all.log`
- `outputs/experiment_3/figure_4/001_contexts_50.json`
- `outputs/experiment_3/figure_7/001_contexts_10.json`
- `outputs/experiment_3/figure_8/001_contexts_60.json`
- `outputs/experiment_3/figure_9/001_contexts_40.json`
- 각 figure의 plot PNG와 `run.log`는 동일 figure 폴더에 저장되어 있다.
