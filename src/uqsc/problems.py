"""Objective functions f(x(theta), theta) = E_v F(x(theta), theta, v).

Arrays follow the convention ``x.shape == (..., q, n)`` and ``theta.shape == (..., n)``,
where q is the output dimension of x and n the number of theta points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Problem:
    q: int
    mu: float
    L: float
    V: float = 0.0
    VG: float = 1.0

    def mean_grad(self, x, theta):
        """grad_x f(x(theta), theta)."""
        raise NotImplementedError

    def grad(self, x, theta, v=None):
        """grad_x F(x(theta), theta, v). Defaults to the noiseless gradient."""
        return self.mean_grad(x, theta)

    def sample_v(self, rng, size):
        """Draw v ~ nu; ``None`` if F does not depend on v."""
        return None

    def optimum(self, theta):
        """x*(theta), shape (q, n), if known in closed form (used for reference coefficients)."""
        raise NotImplementedError

    @property
    def kappa(self) -> float:
        return self.L / self.mu


def crepey_target(theta):
    """Equation (6.1): the optimum used in Crepey et al. and Section 6."""
    s = np.sin(theta)
    return np.abs(0.8 + 0.25 * np.exp(s) - np.cosh(s**2)) * (1 + np.sin(2 * theta))


@dataclass
class QuadraticProblem(Problem):
    """Equation (6.2): f = mu/2 (x - x*)^2 + L/2 (y - x*)^2, with q = 2 components (x, y)."""

    mu: float = 1.0
    L: float = 200.0
    target: callable = crepey_target
    q: int = 2

    def mean_grad(self, x, theta):
        scale = np.array([[self.mu], [self.L]])
        return scale * (x - self.target(theta)[..., None, :])

    def optimum(self, theta):
        t = self.target(theta)
        return np.stack([t, t])


@dataclass
class NoisyQuadraticProblem(QuadraticProblem):
    """Equation (6.3): F = f + v (x + y), v ~ U[-1, 1]. Gives V = 2/3, V_G = 1."""

    V: float = 2.0 / 3.0
    VG: float = 1.0

    def grad(self, x, theta, v=None):
        g = self.mean_grad(x, theta)
        return g if v is None else g + v[..., None, :]

    def sample_v(self, rng, size):
        return rng.uniform(-1.0, 1.0, size=size)
