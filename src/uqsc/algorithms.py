"""Algorithm 3.1 (UQ gradient descent) and Algorithm 5.1 (UQ accelerated gradient descent).

Iterates are the basis coefficients u of x(theta) = sum_i u_i B_i(theta), stored as arrays
of shape (T, q, m_max): T independent trials, q output components, and all coefficients
up to the largest truncation level reached. Entries at or above the current level m_k are
kept at zero, which is the truncation step of both algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .basis import Basis
from .problems import Problem

Schedule = Callable[[int], int]
StepSize = float | Callable[[int, int], float]


@dataclass
class Result:
    u: np.ndarray
    """Final coefficients u^{K+1}, shape (T, q, m_K)."""
    levels: np.ndarray
    """Truncation level m_k used at each iteration k = 1..K."""
    errors: np.ndarray | None
    """||u^{k+1} - P_{m_k}(u*)||_2^2 per trial and iteration, shape (T, K), if a reference was given."""

    @property
    def mean_error(self) -> np.ndarray:
        return self.errors.mean(axis=0)


# ---------------------------------------------------------------------------
# Schedules and step sizes
# ---------------------------------------------------------------------------


def power_schedule(power: float = 0.5, offset: int = 10, shift: int = 2) -> Schedule:
    """m_k = floor((k + offset)^power) + shift."""
    return lambda k: int(np.floor((k + offset) ** power)) + shift


def sqrt_schedule(offset: int = 10, shift: int = 2) -> Schedule:
    """m_k = floor(sqrt(k + offset)) + shift, the schedule of Figures 1 and 2."""
    return power_schedule(0.5, offset, shift)


def constant_schedule(m: int) -> Schedule:
    return lambda k: m


def gd_step_size(problem: Problem, basis: Basis, estimator) -> Callable[[int, int], float]:
    """Maximal step size of Theorem 4.3, gamma_k = 2 / ((mu + L) C_G), with C_G from Corollary 3.5.

    For an exact gradient C_G = 1; for a Monte Carlo estimate with M_k samples and P inner
    samples, C_G = 1 + 2 V_G_hat Q_{m_k} / M_k with V_G_hat = 1 + (V_G - 1) / P.
    """
    P = getattr(estimator, "n_inner", 1)
    VG_hat = 1 + (problem.VG - 1) / P

    def step(k, m):
        M = estimator.n_samples(k, m)
        CG = 1 + 2 * VG_hat * basis.Q(m) / M
        return 2.0 / ((problem.mu + problem.L) * CG)

    return step


def sa_step_size(gamma0: float = 1e-2) -> Callable[[int, int], float]:
    """Stochastic approximation step of Crepey et al.: starts at gamma0 and decays like 1/k."""
    return lambda k, m: 1.0 / (1.0 / gamma0 + k - 1)


def agd_parameters(mu: float, alpha: float) -> tuple[float, float]:
    """(alpha, beta) with beta = (1 - sqrt(alpha mu)) / (1 + sqrt(alpha mu)), as in Theorem 5.1."""
    r = np.sqrt(alpha * mu)
    return alpha, (1 - r) / (1 + r)


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------


def _setup(m_schedule, n_iter, q, n_trials, u1):
    levels = np.array([m_schedule(k) for k in range(1, n_iter + 1)])
    if np.any(np.diff(levels) < 0) or levels[0] < 1:
        raise ValueError("truncation levels m_k must be positive and non-decreasing")
    m_max = int(levels.max())
    u = np.zeros((n_trials, q, m_max))
    if u1 is not None:
        u1 = np.asarray(u1, dtype=float).reshape(q, -1)
        n = min(u1.shape[1], levels[0])
        u[:, :, :n] = u1[:, :n]
    return levels, u


def _error(u, reference, m):
    diff = u.copy()
    diff[:, :, :m] -= reference[:, :m]
    return np.sum(diff**2, axis=(1, 2))


def _as_step(step):
    return step if callable(step) else (lambda k, m: step)


def uq_gradient_descent(
    estimator,
    m_schedule: Schedule,
    n_iter: int,
    step_size: StepSize,
    q: int,
    u1: np.ndarray | None = None,
    n_trials: int = 1,
    reference: np.ndarray | None = None,
    seed: int | None = None,
) -> Result:
    """Algorithm 3.1.

    Each iteration truncates u^k to the first m_k coefficients, estimates D_{m_k} f(x^k)
    and sets u^{k+1} = u^k - gamma_k D'_{m_k} f(x^k).

    ``reference`` holds the coefficients of the true optimum u*, shape (q, >= max m_k); if
    given, the error ||u^{k+1} - P_{m_k}(u*)||^2 is recorded at every iteration.
    """
    rng = np.random.default_rng(seed)
    step_size = _as_step(step_size)
    levels, u = _setup(m_schedule, n_iter, q, n_trials, u1)
    errors = np.empty((n_trials, n_iter)) if reference is not None else None

    for k, m in enumerate(levels, start=1):
        g = estimator.estimate(u[:, :, :m], m, k, rng)
        u[:, :, :m] -= step_size(k, m) * g
        if errors is not None:
            errors[:, k - 1] = _error(u, reference, m)

    return Result(u=u[:, :, : levels[-1]], levels=levels, errors=errors)


def uq_accelerated_gradient_descent(
    estimator,
    m_schedule: Schedule,
    n_iter: int,
    alpha: StepSize,
    beta: StepSize,
    q: int,
    u1: np.ndarray | None = None,
    n_trials: int = 1,
    reference: np.ndarray | None = None,
    seed: int | None = None,
) -> Result:
    """Algorithm 5.1.

    With u^0 = u^1, each iteration truncates to m_k, forms y^k = (1 + beta) u^k - beta u^{k-1},
    estimates D_{m_k} f(z^k) at z^k = I^{-1}(y^k) and sets u^{k+1} = y^k - alpha D'_{m_k} f(z^k).
    """
    rng = np.random.default_rng(seed)
    alpha, beta = _as_step(alpha), _as_step(beta)
    levels, u = _setup(m_schedule, n_iter, q, n_trials, u1)
    u_prev = u.copy()
    errors = np.empty((n_trials, n_iter)) if reference is not None else None

    for k, m in enumerate(levels, start=1):
        b = beta(k, m)
        y = (1 + b) * u[:, :, :m] - b * u_prev[:, :, :m]
        g = estimator.estimate(y, m, k, rng)
        u_prev[:, :, :m] = u[:, :, :m]
        u[:, :, :m] = y - alpha(k, m) * g
        if errors is not None:
            errors[:, k - 1] = _error(u, reference, m)

    return Result(u=u[:, :, : levels[-1]], levels=levels, errors=errors)


def stochastic_approximation(estimator, m_schedule, n_iter, q, gamma0=1e-2, **kwargs) -> Result:
    """Baseline of Crepey et al. [6]: Algorithm 3.1 with decaying steps gamma_k ~ 1/k."""
    return uq_gradient_descent(estimator, m_schedule, n_iter, sa_step_size(gamma0), q, **kwargs)
