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
    """||u^{k+1} - P_{m_k}(u*)||_2^2 per trial at the recorded iterations, if a reference was given."""
    iterations: np.ndarray | None = None
    """Iterations k (1-based) at which ``errors`` were recorded; None means every iteration."""

    @property
    def mean_error(self) -> np.ndarray:
        return self.errors.mean(axis=0)

    @property
    def recorded_levels(self) -> np.ndarray:
        """m_k at the recorded iterations."""
        return self.levels if self.iterations is None else self.levels[self.iterations - 1]


class _Recorder:
    """Records ||u - P_m(u*)||^2 at every iteration, or only at the iterations in ``record_at``.

    Recording at a few hundred log-spaced iterations keeps memory and time small for runs
    of millions of SGD steps.
    """

    def __init__(self, n_iter, n_trials, reference, record_at=None):
        self.reference = reference
        if reference is None:
            self.iterations, self.errors = None, None
            return
        self.iterations = None if record_at is None else np.unique(np.clip(np.asarray(record_at), 1, n_iter))
        n_rec = n_iter if self.iterations is None else len(self.iterations)
        self.errors = np.full((n_trials, n_rec), np.nan)
        self._next = 0

    def __call__(self, k, u, m):
        if self.reference is None:
            return
        if self.iterations is None:
            self.errors[:, k - 1] = _error(u, self.reference, m)
        elif self._next < len(self.iterations) and self.iterations[self._next] == k:
            self.errors[:, self._next] = _error(u, self.reference, m)
            self._next += 1


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
    record_at: np.ndarray | None = None,
) -> Result:
    """Algorithm 3.1.

    Each iteration truncates u^k to the first m_k coefficients, estimates D_{m_k} f(x^k)
    and sets u^{k+1} = u^k - gamma_k D'_{m_k} f(x^k).

    ``reference`` holds the coefficients of the true optimum u*, shape (q, >= max m_k); if
    given, the error ||u^{k+1} - P_{m_k}(u*)||^2 is recorded at every iteration, or only at
    the iterations in ``record_at``.
    """
    rng = np.random.default_rng(seed)
    step_size = _as_step(step_size)
    levels, u = _setup(m_schedule, n_iter, q, n_trials, u1)
    record = _Recorder(n_iter, n_trials, reference, record_at)

    for k, m in enumerate(levels, start=1):
        g = estimator.estimate(u[:, :, :m], m, k, rng)
        u[:, :, :m] -= step_size(k, m) * g
        record(k, u, m)

    return Result(u=u[:, :, : levels[-1]], levels=levels, errors=record.errors, iterations=record.iterations)


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


def adaptive_uq_descent(
    problem: Problem,
    basis: Basis,
    estimator,
    m_start: int,
    m_max: int,
    n_iter: int,
    step_size: StepSize | None = None,
    alpha: StepSize | None = None,
    beta: StepSize | None = None,
    check_every: int = 10,
    tol: float = 1e-3,
    grow: Callable[[int], int] | None = None,
    n_trials: int = 1,
    reference: np.ndarray | None = None,
    n_quad: int = 2**12,
    seed: int | None = None,
    record_at: np.ndarray | None = None,
) -> Result:
    """Algorithm 3.1 (``step_size``) or Algorithm 5.1 (``alpha``, ``beta``) with the stall rule of ALSP.

    Instead of a fixed schedule, the basis grows from m to ``grow(m)`` (default m + 1)
    whenever the objective, evaluated by quadrature every ``check_every`` iterations and
    averaged over trials, has decreased by less than a relative ``tol`` since the last check
    (Dong et al., Algorithm 1). m_k then depends on the iterates, which the analysis of
    Sections 4 and 5 does not cover. Step sizes may depend on (k, m); if ``beta`` is omitted
    it is set from alpha as in Theorem 5.1. A run that diverges is stopped, and its
    remaining errors are recorded as NaN.
    """
    accelerated = alpha is not None
    if accelerated == (step_size is not None):
        raise ValueError("pass either step_size (gradient descent) or alpha and beta (accelerated)")
    rng = np.random.default_rng(seed)
    grow = grow or (lambda m: m + 1)
    if accelerated:
        alpha = _as_step(alpha)
        beta = _as_step(beta) if beta is not None else (lambda k, m: agd_parameters(problem.mu, alpha(k, m))[1])
    else:
        step_size = _as_step(step_size)
    nodes, weights = basis.quadrature(n_quad)
    B = basis.evaluate(nodes, m_max)
    u = np.zeros((n_trials, problem.q, m_max))
    u_prev = u.copy()
    levels = np.empty(n_iter, dtype=int)
    record = _Recorder(n_iter, n_trials, reference, record_at)
    m, last = m_start, np.inf

    for k in range(1, n_iter + 1):
        if accelerated:
            b = beta(k, m)
            y = (1 + b) * u[:, :, :m] - b * u_prev[:, :, :m]
            g = estimator.estimate(y, m, k, rng)
            u_prev[:, :, :m] = u[:, :, :m]
            u[:, :, :m] = y - alpha(k, m) * g
        else:
            g = estimator.estimate(u[:, :, :m], m, k, rng)
            u[:, :, :m] -= step_size(k, m) * g
        levels[k - 1] = m
        if not np.all(np.isfinite(u)) or np.abs(u).max() > 1e8:
            levels[k - 1:] = m  # errors from here on stay NaN
            break
        record(k, u, m)
        if k % check_every == 0:
            obj = float(np.mean(problem.value(u[:, :, :m] @ B[:m], nodes) @ weights))
            if last - obj < tol * abs(obj) and m < m_max:
                m = min(m_max, grow(m))
                obj = np.inf
            last = obj

    return Result(u=u[:, :, :m], levels=levels, errors=record.errors, iterations=record.iterations)
