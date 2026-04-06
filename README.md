# SIR Parameter Estimation (Toy Model)

SIR 감염병 모델의 파라미터 추정을 세 가지 방법으로 비교하는 토이 프로젝트입니다.

- **Least Squares (최소제곱법)**
- **Maximum Likelihood Estimation, MLE (최대우도추정)**
- **Bayesian Inference (베이지안 추론)**

---

## SIR 모델

SIR 모델은 전체 인구 $N = S + I + R$을 세 구획으로 나눕니다.

$$\frac{dS}{dt} = -\frac{\beta S I}{N}, \quad
\frac{dI}{dt} = \frac{\beta S I}{N} - \gamma I, \quad
\frac{dR}{dt} = \gamma I$$

| 파라미터 | 의미 |
|---|---|
| $\beta$ | 감염률 (infection rate) |
| $\gamma$ | 회복률 (recovery rate) |
| $R_0 = \beta / \gamma$ | 기초감염재생산수 |

**True parameters (이 프로젝트에서 사용):**

| | 값 |
|---|---|
| $\beta$ | 0.3 |
| $\gamma$ | 0.1 |
| $R_0$ | 3.0 |
| $N = S_0 + I_0 + R_0$ | 1000 |

---

## 관측 모델

감염자 수 $I(t)$만 관측 가능하며, 가우시안 노이즈가 추가됩니다.

$$I_{\text{obs}}(t) = I_{\text{model}}(t;\, \beta, \gamma) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2), \quad \sigma = 10$$

---

## 추정 방법 비교

### 1. Least Squares

잔차 제곱합(SSE)을 최소화합니다.

$$\hat{\theta}_{\text{LS}} = \arg\min_{\beta, \gamma} \sum_t \left( I_{\text{obs}}(t) - I_{\text{model}}(t;\, \beta, \gamma) \right)^2$$

최적화 알고리즘: L-BFGS-B

### 2. Maximum Likelihood Estimation

가우시안 노이즈 가정 하에서 음의 로그우도를 최소화합니다.

$$\hat{\theta}_{\text{MLE}} = \arg\min_{\beta, \gamma} \sum_t \left[ \frac{(I_{\text{obs}}(t) - I_{\text{model}}(t))^2}{2\sigma^2} + \log(\sqrt{2\pi}\,\sigma) \right]$$

> **참고:** 가우시안 노이즈 + 고정된 $\sigma$ 조건에서 MLE는 LS와 동일한 해를 가집니다.

### 3. Bayesian Inference

사전분포와 우도를 결합하여 사후분포를 계산합니다.

$$p(\beta, \gamma \mid \text{data}) \propto p(\text{data} \mid \beta, \gamma)\, p(\beta, \gamma)$$

- **Prior:** $\beta \sim \text{Uniform}(0, 1)$, $\gamma \sim \text{Uniform}(0, 1)$
- **Likelihood:** 가우시안 관측 모델
- **계산 방법:** 2D 그리드에서 직접 사후분포 계산 (grid-based)

요약 통계량:
- **MAP (Maximum A Posteriori):** 사후분포를 최대화하는 점 추정값
- **Posterior Mean:** 사후분포의 기댓값

---

## 추정 결과

| 방법 | $\hat{\beta}$ | $\hat{\gamma}$ |
|---|---|---|
| True value | 0.3000 | 0.1000 |
| Least Squares | 0.2998 | 0.1007 |
| MLE | 0.2998 | 0.1007 |
| Bayesian MAP | 0.2976 | 0.1002 |
| Bayesian Mean | 0.2983 | 0.1005 |

---

## 프로젝트 구조

```
bayesian/
├── src/
│   ├── model.py          # SIR ODE 정의 및 수치해석 (solve_sir)
│   └── estimation.py     # LS / MLE / Bayesian 추정 함수
├── notebooks/
│   ├── 01_generate_data.ipynb   # 합성 데이터 생성 및 저장
│   ├── 02_least_squares.ipynb   # LS 파라미터 추정
│   ├── 03_mle.ipynb             # MLE 파라미터 추정
│   └── 04_bayesian.ipynb        # 베이지안 사후분포 계산 및 시각화
├── data/
│   └── synthetic_sir.csv        # 노이즈가 추가된 합성 관측 데이터
├── requirements.txt
└── README.md
```

### 주요 함수 (`src/estimation.py`)

| 함수 | 설명 |
|---|---|
| `estimate_parameters_ls()` | L-BFGS-B로 SSE 최소화 |
| `estimate_parameters_mle()` | L-BFGS-B로 NLL 최소화 |
| `compute_posterior_grid()` | 2D 그리드에서 사후분포 계산 (log-sum-exp 안정화 포함) |
| `map_estimate_from_grid()` | 사후분포 최대값 위치 반환 |
| `posterior_mean_from_grid()` | 사후분포의 기댓값 계산 |

---

## 환경 설정

Python 3.9 이상 권장

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**핵심 의존성:**

| 패키지 | 용도 |
|---|---|
| `numpy` | 수치 연산 |
| `scipy` | ODE 수치해석 (`odeint`), 최적화 (`minimize`) |
| `matplotlib` | 시각화 |
| `pandas` | 데이터 저장/로드 |
| `jupyter` / `jupyterlab` | 노트북 실행 |

---

## 실행 순서

노트북을 순서대로 실행합니다.

```bash
jupyter lab
```

1. `01_generate_data.ipynb` — 합성 데이터 생성 → `data/synthetic_sir.csv` 저장
2. `02_least_squares.ipynb` — LS 추정
3. `03_mle.ipynb` — MLE 추정
4. `04_bayesian.ipynb` — 베이지안 사후분포 계산 및 시각화

> 노트북은 프로젝트 루트를 `sys.path`에 추가하므로, 반드시 `notebooks/` 디렉토리 내에서 실행해야 합니다.

---

## 학습 목적

이 프로젝트의 목표는 세 추정 방법의 차이를 직접 확인하는 것입니다.

| 관점 | LS | MLE | Bayesian |
|---|---|---|---|
| 출력 | 점 추정값 | 점 추정값 | 사후분포 전체 |
| 불확실성 정량화 | 없음 | 없음 (별도 계산 필요) | 자연스럽게 포함 |
| 사전 정보 활용 | 없음 | 없음 | 사전분포로 반영 |
| 가우시안 노이즈 하에서 | MLE와 동일 | LS와 동일 | MAP ≈ MLE (uniform prior) |
| 확장성 | 노이즈 모델 변경 어려움 | 우도 함수 교체 가능 | 우도 + 사전분포 모두 교체 가능 |
