"""Estimators of the truncated gradient D_m f(x) (Definition 2.5).

``estimate(u, m, k, rng)`` takes coefficients ``u`` of shape (T, q, m) -- T independent
trials run in parallel -- and returns the (possibly noisy) gradient coefficients,
also of shape (T, q, m).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .basis import Basis
from .problems import Problem

SampleSchedule = int | Callable[[int, int], int]


class ExactGradient:
    """D_m f computed by a fixed high-accuracy quadrature rule for pi (noiseless)."""

    def __init__(self, problem: Problem, basis: Basis, n_quad: int = 2**14):
        self.problem, self.basis = problem, basis
        self.nodes, self.weights = basis.quadrature(n_quad)
        self._B = np.empty((0, len(self.weights)))

    def _basis(self, m):
        if self._B.shape[0] < m:
            self._B = self.basis.evaluate(self.nodes, max(m, 2 * self._B.shape[0]))
        return self._B[:m]

    def n_samples(self, k, m):
        return np.inf

    def estimate(self, u, m, k, rng):
        B = self._basis(m)
        x = u @ B
        g = self.problem.mean_grad(x, self.nodes)
        return (g * self.weights) @ B.T


class MonteCarloGradient:
    """Monte Carlo estimate of Equation (3.5).

    At iteration k with level m, draws M_k samples theta_j ~ pi and, for each,
    P samples v_{j,p} ~ nu, and averages grad F(x(theta_j), theta_j, v_{j,p}) B_i(theta_j).
    ``n_samples`` may be an int or a callable ``(k, m) -> M_k``.
    """

    def __init__(self, problem: Problem, basis: Basis, n_samples: SampleSchedule = 500, n_inner: int = 1):
        self.problem, self.basis = problem, basis
        self._M = n_samples
        self.n_inner = n_inner

    def n_samples(self, k, m):
        return self._M(k, m) if callable(self._M) else self._M

    def estimate(self, u, m, k, rng):
        T = u.shape[0]
        M = self.n_samples(k, m)
        theta = self.basis.sample(rng, (T, M))
        B = self.basis.evaluate(theta, m)  # (T, m, M)
        x = np.einsum("tqm,tmn->tqn", u, B)
        g = np.zeros_like(x)
        for _ in range(self.n_inner):
            g += self.problem.grad(x, theta, self.problem.sample_v(rng, (T, M)))
        g /= self.n_inner
        return np.einsum("tqn,tmn->tqm", g, B) / M
