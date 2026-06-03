# Federated Learning Over Wireless Networks Reproduction

이 저장소는 Chen et al., "A Joint Learning and Communications Framework for
Federated Learning Over Wireless Networks"의 실험을 재현하기 위한 코드입니다.
Fig. 3-10은 구현 대상이 아니라 검증 기준이며, 목표는 논문에 기술된 wireless FL
실험 절차를 실제로 수행하는 것입니다.

## 구현 원칙

- 논문에 명시된 시스템/학습 파라미터는 `main.py`에서 중앙 관리합니다.
- 논문에 명시되지 않은 실험 선택값은 `configs/*.yaml`에 명시적으로 둡니다.
- 로컬 학습은 Gradient Descent 기반으로 수행합니다.
- 결과 그래프는 `outputs/` 아래에 저장합니다.
- `blueprints/main.py`에 정의된 파일/함수 범위 안에서 구현합니다.

## 구조

- `main.py`: 데이터 준비, wireless link 계산, FL round, baseline, proposed algorithm, figure runner.
- `blueprints/main.py`: 허용된 public 함수/클래스 구조.
- `configs/figure_*.yaml`: Fig. 3-10별 sweep/config 입력.
- `docs/`: 논문 PDF와 원본 MATLAB Wireless-FL 참고 코드.
- `outputs/`: 실행 결과 그래프. git에는 기본적으로 포함하지 않습니다.

## 환경

Python 3.10 이상을 권장합니다. 주요 의존성은 다음과 같습니다.

```bash
pip install numpy scipy matplotlib torch pyyaml scikit-learn torchvision
```

MNIST는 다음 순서로 로드합니다.

1. `data/mnist.npz`
2. `mnist.npz`
3. `/home/imes-server6/dataset/mnist.npz`
4. torchvision MNIST cache 또는 명시적 download fallback
5. 마지막 fallback: scikit-learn digits. 이 fallback은 논문 MNIST 재현이 아닙니다.

## 실행

단일 figure 실행:

```bash
python main.py --figure 9 --config configs/figure_9.yaml --output-dir outputs --plot
```

전체 figure 실행:

```bash
python main.py --figure all --output-dir outputs --plot
```

상세 round 로그가 필요하면 `--verbose`를 추가합니다. 로그는
`outputs/verbose_logs/` 아래에 저장됩니다.

## Fig. 9 baseline c 모드

`configs/figure_9.yaml`의 `training.baseline_c_mode`로 baseline c 구현을 선택합니다.

- `current`: packet error rate만을 비용으로 둔 Hungarian assignment.
- `wireless_fl`: 원본 MATLAB `docs/Wireless-FL/FLMIN.m`의 baseline 3 방식. Munkres로
  wireless-only 사용자 집합을 고른 뒤 RB assignment를 랜덤화합니다.

최근 Fig. 9 모드 비교는 seed `0..9`, RB `[3, 6, 9, 12]`, MNIST test `10000`,
`130` rounds, learning rate `0.08` 조건에서 수행했습니다.

| RBs | Proposed | Baseline a | Baseline b | Baseline c current | Baseline c wireless_fl |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.84735 | 0.82920 | 0.78902 | 0.84078 | 0.78884 |
| 6 | 0.85087 | 0.82301 | 0.81148 | 0.84849 | 0.81400 |
| 9 | 0.85295 | 0.84109 | 0.82541 | 0.85195 | 0.82343 |
| 12 | 0.85372 | 0.84343 | 0.84846 | 0.85337 | 0.84034 |

`wireless_fl` 모드가 논문 Fig. 9의 baseline c 격차를 더 잘 반영합니다.

## 검증

구문 검사는 다음 명령으로 수행했습니다.

```bash
python -m py_compile main.py
```

Fig. 9 10-seed 비교 그래프는 다음 경로에 생성되었습니다.

- `outputs/figure_9_mode_compare/figure_9_current_seeds_0_9.png`
- `outputs/figure_9_mode_compare/figure_9_wireless_fl_seeds_0_9.png`
- `outputs/figure_9_mode_compare/figure_9_baseline_c_mode_compare_seeds_0_9.png`
