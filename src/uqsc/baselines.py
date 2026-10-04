"""Baselines for the cost comparison of Section 6.

* ``learn_solution_path``: LSP of Dong, Grigas & Gupta (AISTATS 2025), i.e. constant-step
  SGD with one sample of theta per step on a fixed basis, with step eta_bar / (C L),
  where C = Q_m for a basis orthonormal with respect to pi (their Theorem 3.4).
* ``adaptive_learn_solution_path``: their ALSP (Algorithm 1), which adds one basis
  function whenever the objective, evaluated by quadrature, stops decreasing.
* ``naive_monte_carlo``: sample theta_j ~ pi, solve each problem by gradient descent,
  and estimate the coefficients of x* by Monte Carlo (the method of Section 1).
"""

from __future__ import annotations

import numpy as np

from .algorithms import Result, _Recorder, constant_schedule, uq_gradient_descent
from .basis import Basis
from .gradients import MonteCarloGradient
from .problems import Problem


def learn_solution_path(problem: Problem, basis: Basis, m: int, n_iter: int, eta_bar: float = 1.0, **kwargs) -> Result:
    est = MonteCarloGradient(problem, basis, n_samples=1)
    step = eta_bar / (basis.Q(m) * problem.L)
    return uq_gradient_descent(est, constant_schedule(m), n_iter, step, q=problem.q, **kwargs)


def adaptive_learn_solution_path(
    problem: Problem,
    basis: Basis,
    m_start: int,
    m_max: int,
    n_iter: int,
    eta_bar: float = 1.0,
    check_every: int = 2000,
    tol: float = 1e-4,
    grow=None,
    n_trials: int = 1,
    reference: np.ndarray | None = None,
    n_quad: int = 2**12,
    seed: int | None = None,
    record_at: np.ndarray | None = None,
) -> Result:
    """ALSP with SGD as the inner solver.

    Every ``check_every`` steps the objective is evaluated by quadrature (not counted as
    gradient evaluations, as in Dong et al.). If it has decreased by less than a relative
    ``tol`` since the last check, the basis grows from m to ``grow(m)`` (default m + 1, one
    basis function as in their Algorithm 1). To keep trials vectorised, the decision uses
    the objective averaged over trials.
    """
    rng = np.random.default_rng(seed)
    grow = grow or (lambda m: m + 1)
    est = MonteCarloGradient(problem, basis, n_samples=1)
    nodes, weights = basis.quadrature(n_quad)
    B = basis.evaluate(nodes, m_max)
    u = np.zeros((n_trials, problem.q, m_max))
    levels = np.empty(n_iter, dtype=int)
    record = _Recorder(n_iter, n_trials, reference, record_at)
    m, last = m_start, np.inf

    for k in range(1, n_iter + 1):
        g = est.estimate(u[:, :, :m], m, k, rng)
        u[:, :, :m] -= eta_bar / (basis.Q(m) * problem.L) * g
        levels[k - 1] = m
        record(k, u, m)
        if k % check_every == 0:
            obj = float(np.mean(problem.value(u[:, :, :m] @ B[:m], nodes) @ weights))
            if last - obj < tol * abs(obj) and m < m_max:
                m = min(m_max, grow(m))
                obj = np.inf
            last = obj

    return Result(u=u[:, :, :m], levels=levels, errors=record.errors, iterations=record.iterations)


def naive_monte_carlo(
    problem: Problem,
    basis: Basis,
    m: int,
    n_samples: int,
    solver_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate the first m coefficients of x* from n_samples solves.

    Each x*(theta_j) is computed by ``solver_iters`` steps of gradient descent with step
    2 / (mu + L), so the cost is n_samples * solver_iters gradient evaluations.
    """
    theta = basis.sample(rng, (n_samples,))
    x = np.zeros((problem.q, n_samples))
    step = 2.0 / (problem.mu + problem.L)
    for _ in range(solver_iters):
        x -= step * problem.mean_grad(x, theta)
    return x @ basis.evaluate(theta, m).T / n_samples
