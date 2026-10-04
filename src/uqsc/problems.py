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

    def value(self, x, theta):
        """f(x(theta), theta), shape (..., n). Only needed by the adaptive baseline."""
        raise NotImplementedError

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


@dataclass
class MarkowitzProblem(Problem):
    """Example 1.1 with an uncertain factor covariance and, optionally, noisy returns.

    f(w, theta) = -w^T r + (lam/2) w^T Sigma(theta) w + (tau/2) ||w - w0||^2 + (alpha/2) (e^T w - 1)^2,

    with Sigma(theta) = diag(D) + sum_j s_j exp(gamma theta_j) b_j b_j^T: each coordinate of
    theta ~ U[-1, 1]^d scales the variance of one risk factor. With ``return_noise > 0``,
    F = f - w^T v with v ~ U[-s, s]^n models estimated returns, so V = n s^2 / 3 and V_G = 1.
    Unlike (6.2), Sigma(theta) couples x(theta) across theta, so the level-m optimum u*_m
    differs from the projection P_m(u*).
    """

    n_assets: int = 10
    n_factors: int = 2
    risk_aversion: float = 2.0
    turnover: float = 0.2
    budget_penalty: float = 0.1
    vol_sensitivity: float = 3.0
    factor_scale: float = 0.05
    return_range: tuple = (0.0, 0.1)
    return_noise: float = 0.0
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        n, d = self.n_assets, self.n_factors
        self.q = n
        self.r = rng.uniform(*self.return_range, n)
        self.w0 = np.full(n, 1.0 / n)
        self.loadings = np.vstack([rng.uniform(0.5, 1.5, n), rng.normal(0.0, 1.0, (d - 1, n))])
        self.factor_var = self.factor_scale * np.linspace(1.0, 0.5, d)
        self.idio_var = rng.uniform(0.02, 0.1, n)
        self.V = n * self.return_noise**2 / 3
        self.VG = 1.0
        # Sigma(theta) is increasing in each theta_j, so the extreme eigenvalues of the
        # Hessian are attained at the corners theta = -1 and theta = +1.
        eig_lo = np.linalg.eigvalsh(self._hessian(-np.ones((d, 1)))[0])
        eig_hi = np.linalg.eigvalsh(self._hessian(np.ones((d, 1)))[0])
        self.mu, self.L = float(eig_lo[0]), float(eig_hi[-1])

    def _factor_scale(self, theta):
        return self.factor_var[:, None] * np.exp(self.vol_sensitivity * theta)

    def _sigma_times(self, x, theta):
        s = self._factor_scale(theta)
        bx = np.einsum("jq,...qn->...jn", self.loadings, x)
        return self.idio_var[:, None] * x + np.einsum("jq,...jn->...qn", self.loadings, s * bx)

    def _hessian(self, theta):
        """lam Sigma(theta) + tau I + alpha e e^T at points theta of shape (d, n), as (n, q, q)."""
        s = self._factor_scale(theta).T
        sigma = np.einsum("nj,jq,jp->nqp", s, self.loadings, self.loadings) + np.diag(self.idio_var)
        return self.risk_aversion * sigma + self.turnover * np.eye(self.q) + self.budget_penalty

    def mean_grad(self, x, theta):
        budget = x.sum(axis=-2, keepdims=True) - 1.0
        return (
            -self.r[:, None]
            + self.risk_aversion * self._sigma_times(x, theta)
            + self.turnover * (x - self.w0[:, None])
            + self.budget_penalty * budget
        )

    def grad(self, x, theta, v=None):
        g = self.mean_grad(x, theta)
        return g if v is None else g - v

    def sample_v(self, rng, size):
        if self.return_noise == 0:
            return None
        size = tuple(np.atleast_1d(size))
        return rng.uniform(-self.return_noise, self.return_noise, size=size[:-1] + (self.q, size[-1]))

    def value(self, x, theta):
        budget = x.sum(axis=-2) - 1.0
        return (
            -np.einsum("q,...qn->...n", self.r, x)
            + 0.5 * self.risk_aversion * np.sum(x * self._sigma_times(x, theta), axis=-2)
            + 0.5 * self.turnover * np.sum((x - self.w0[:, None]) ** 2, axis=-2)
            + 0.5 * self.budget_penalty * budget**2
        )

    def optimum(self, theta):
        rhs = self.r + self.turnover * self.w0 + self.budget_penalty
        return np.linalg.solve(self._hessian(theta), rhs[None, :, None])[..., 0].T
