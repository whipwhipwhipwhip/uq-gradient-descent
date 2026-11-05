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

## Reproducing the paper

```bash
uv run uqsc-paper              # every figure and table in Section 6, into results/paper/
uv run uqsc-paper fig1 table3  # only some of them
uv run uqsc-paper --list       # what each item is
```

`uqsc-paper` runs each experiment with the settings used in the paper and writes the
files for LaTeX under their number in the paper. PNG previews, the plotted curves (`.npz`)
and intermediate files go to `results/paper/raw/`, under the experiment names used by
`uqsc-experiments` (below).

| Item | Writes | Experiment | Settings |
|---|---|---|---|
| `fig1` | `fig1a.pdf`, `fig1b.pdf` | `quadratic_exact`, `quadratic_mc` | Quadratic (6.2)/(6.3), µ=1, L=200, m_k = ⌊√(k+10)⌋ + 2, M=500, 200 trials |
| `fig2` | `fig2.pdf` | `fixed_vs_growing` | m_k = ⌊(k+10)^0.7⌋ + 2 vs. fixed m = 91, M=250, 600 iterations, 200 trials |
| `fig3` | `fig3.pdf` | `markowitz` | Portfolio problem with noisy returns, M_k = 2Q_{m_k}, 20 trials |
| `fig4` (= `table1`) | `fig4.pdf`, `table1.tex` | `markowitz_uq` | One UQ-AGD run (600 iterations) against 20 000 Monte Carlo samples of θ |
| `fig5` | `fig5.pdf` | `comparison` | UQ-GD, UQ-AGD, LSP (m = 10/28/66), ALSP, naive MC; V = 0, 20 trials |
| `fig6` (= `table2`) | `fig6.pdf`, `table2.tex` | `logistic_kappa`, `logistic_alsp_tune` | See below |
| `table3` | `table3.tex` | `schedules` | One total degree every 25/75/200 iterations; V = 0, 20 trials |

Every table is also written as `.md`. All runs use seed 0, and with the package versions
pinned in `uv.lock` they reproduce the paper's figures and tables exactly. Other versions of
NumPy or its linear algebra backend may change the results in the last digits.

Runtimes of `uv run uqsc-paper` on an Apple M4 Max (14-core CPU) with 36 GB of
RAM.

| Item | Runtime |
|---|---|
| `fig1` | 14 s |
| `fig2` | 31 s |
| `fig3` | 2 s |
| `fig4` | 1 s |
| `fig5` | 13 min |
| `fig6` (= `table2`) | 2 h 25 min |
| `table3` | 7 min |
| All | 2 h 45 min |

**Fig. 6 / Table 2 protocol** (Section 6.3.1). At κ ≈ 16, 60 and 236, ALSP's check
interval (250/1000/4000) and the stall window of UQ-AGD (2/5) are tuned on 5 trials each,
and the selected settings are rerun with 20 trials. At κ ≈ 940 the settings from κ ≈ 236
are reused, with 5 trials and an ALSP budget of 6 Q_{m_max} κ steps. ALSP alone is then
re-tuned over the carried-over check interval (4000), 16000 and 64000. The data are the
LSAC subsample published by Dong et al., downloaded on first use from their repository at
a pinned commit and cached in `~/.cache/uqsc` (override with `UQSC_CACHE`).

## Data

The logistic regression of Fig. 6 and Table 2 uses the 1000-row subsample of the LSAC
National Longitudinal Bar Passage Study (L. F. Wightman, LSAC Research Report Series, Law
School Admission Council, 1998), as processed by Q. Dong, P. Grigas and V. Gupta for
*Beyond Discretization: Learning the Optimal Solution Path* (AISTATS 2025). The files are
downloaded from their repository,
[Cumberkid/Learning-the-Optimal-Solution-Path](https://github.com/Cumberkid/Learning-the-Optimal-Solution-Path)
(MIT License, Copyright (c) 2018 paulgrigas), at commit `cf499df`, and are not
redistributed here. LSP and ALSP in `baselines.py` are our implementations of their methods.

## Other experiments

`uqsc-experiments` runs any experiment by name, including the ones not in the paper:

```bash
uv run uqsc-experiments lemma31 ablation_batch1 --cost-trials 20 --paper
```

Outputs go to `results/` (`--out` to change). `--paper` also writes `.pdf` versions in
8 pt Computer Modern without titles, sized for LaTeX.

| Name | What it shows |
|---|---|
| `lemma31` | ‖P_m(u\*) − u\*_m‖² against ‖R_m‖² and the (κ−1)²/(4κ)‖R_m‖² bound of Lemma 3.1 |
| `comparison_adaptive` | The Fig. 5 comparison with ALSP's stall rule used by both UQ methods, which isolates the effect of acceleration |
| `ablation_batch1` | Accelerated SGD at batch size 1 diverges: acceleration needs M_k to grow with Q_m |
| `ablation_increment` | The stall rule adding one function vs. a whole total degree per step, for ALSP and both UQ methods |
| `schedules` | The curves behind Table 3, as a two-panel figure |

**Tuning ALSP.** `alsp_sweep` and `sweep_extended` sweep the stall-rule settings of ALSP
and of the adaptive UQ methods on the portfolio problem, and write tables of the gradient
evaluations each setting needs (`alsp_sweep.md`, `sweep_extended.md`), marking the setting
selected by the rule fixed in advance (fewest evaluations to 1e-6, ties at 1e-8). The
selected settings are recorded in `ALSP_CHECK_EVERY`, `ALSP_TOL` and `STALL` in
`experiments.py`, which Fig. 5 and the ablations use. To change them, edit those constants
and rerun the figures.

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

- `basis.py`: `TrigBasis` (U[−π,π]), `LegendreBasis` (U[−1,1]) and
  `TensorLegendreBasis` (U[−1,1]^d, ordered by total degree). Each provides sampling,
  quadrature and Q_m.
- `problems.py`: the `Problem` interface (`mean_grad`, `grad`, `sample_v`) and the
  quadratics (6.2) and (6.3), `MarkowitzProblem` (Example 1.1) and
  `WeightedLogisticProblem` (6.6). Subclass `Problem` to add new objectives.
- `gradients.py`: `ExactGradient` (quadrature) and `MonteCarloGradient`, i.e.
  Equation (3.5), with M_k samples of θ and `n_inner` = P samples of v. M_k may be a
  schedule `(k, m) -> int`.
- `baselines.py`: Dong et al.'s LSP and ALSP, and naive per-sample Monte Carlo.
- `paper.py`: the `uqsc-paper` command. It maps each figure and table of the paper to an
  experiment and its settings.
- `experiments.py`: every experiment, and the `uqsc-experiments` command.
- `data.py`: downloads and caches the logistic regression data of Dong et al.
- `algorithms.py`: both algorithms, their stall-rule variant `adaptive_uq_descent`, the SA
  baseline, the step-size rules and the schedules. Step sizes may be constants or callables `(k, m) -> float`. All runs are
  vectorised over `n_trials` independent trials.

## License

MIT, see [LICENSE](LICENSE). The data used by Fig. 6 are covered by the license of their
source repository (see [Data](#data)).
