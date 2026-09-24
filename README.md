# uqsc

Implementation of *Uncertainty Quantification for the Gradient and Accelerated Gradient
Descent Methods on Strongly Convex Functions*, a paper I wrote during my PhD.

The optimum x\*(θ) of a strongly convex f(x(θ), θ) is expanded in an orthonormal basis of
L²_π, x(θ) = Σᵢ uᵢ Bᵢ(θ), and the coefficients u are computed by first-order methods
whose truncation level m_k grows over the iterations.

## Setup

```bash
uv sync --extra dev        # or: pip install -e ".[dev]"
uv run pytest
```

## Reproducing Section 6

```bash
uv run uqsc-experiments all                 # all figures, 200 trials each (~45 s)
uv run uqsc-experiments fig1b --trials 50   # a single figure
```

Figures (`.png`) and the plotted curves (`.npz`) are written to `results/`. Add
`--paper` to also write `.pdf` versions for LaTeX. These are sized for half the text
width (3.2 × 2.4 in, 8 pt Computer Modern), have no title, and should be included with
`\includegraphics[width=\linewidth]{fig1a.pdf}` inside a `0.48\textwidth` subfigure.

| Figure | Setting |
|---|---|
| `fig1a` | Quadratic (6.2), µ=1, L=200, exact gradients. GD with γ = 2/(µ+L), AGD with α = 1/L, β = (1−√(αµ))/(1+√(αµ)). |
| `fig1b` | Same, but Monte Carlo gradients with M=500. GD uses γ_k = 2/((µ+L)C_G) with C_G = 1 + 2Q_{m_k}/M. AGD keeps α = 1/L. SGD baseline (Crépey et al.) uses γ_k = 1/(100 + k − 1). |
| `fig2`  | Noisy quadratic (6.3), v ~ U[−1,1], GD with the Fig. 1b step sizes. |
| `fig3`  | µ=1, L=200, M=250, 600 iterations, m_k = ⌊(k+10)^0.7⌋ + 2: UQ (growing m_k) vs. a fixed level m = 91. |

Figures 1 and 2 use m_k = ⌊√(k+10)⌋ + 2. All figures use the trigonometric basis on
θ ~ U[−π, π], the target x\* of (6.1), and the error ‖u^k − P_{m_k}(u\*)‖². The
reference u\* comes from trapezoidal quadrature with 2¹⁴ points.

**Note on Fig. 3:** the schedule's power is 0.7. With it, m_k reaches exactly 91 at k = 600, which is the fixed level and
the C_G = 1 + 182/250 stated in §6.1. The fixed level defaults to the largest m_k the
schedule reaches; override it with `--fixed-m` and the power with `--power`.

## Library

```python
from uqsc import *

problem, basis = QuadraticProblem(mu=1, L=200), TrigBasis()
est = MonteCarloGradient(problem, basis, n_samples=500)       # or ExactGradient(problem, basis)
ref = basis.coefficients(problem.optimum, 30)                 # optional, for error tracking

gd = uq_gradient_descent(est, sqrt_schedule(), 300, gd_step_size(problem, basis, est),
                         q=problem.q, n_trials=10, reference=ref)          # Algorithm 3.1
alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
agd = uq_accelerated_gradient_descent(est, sqrt_schedule(), 300, alpha, beta,
                                      q=problem.q, reference=ref)          # Algorithm 5.1

u = agd.u[0]                         # (q, m) coefficients; mean of x*(θ) is u[:, 0],
var = (u[:, 1:] ** 2).sum(axis=1)    # variance is the sum of the remaining squared coefficients
```

- `basis.py`: `TrigBasis` (U[−π,π]) and `LegendreBasis` (U[−1,1]). Each provides
  sampling, quadrature and Q_m.
- `problems.py`: the `Problem` interface (`mean_grad`, `grad`, `sample_v`) and the
  quadratics (6.2) and (6.3). Subclass `Problem` to add new objectives.
- `gradients.py`: `ExactGradient` (quadrature) and `MonteCarloGradient`, i.e.
  Equation (3.5), with M_k samples of θ and `n_inner` = P samples of v. M_k may be a
  schedule `(k, m) -> int`.
- `algorithms.py`: both algorithms, the SA baseline, the step-size rules and the
  schedules. Step sizes may be constants or callables `(k, m) -> float`. All runs are
  vectorised over `n_trials` independent trials.
