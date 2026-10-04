"""Orthonormal bases of L^2_pi, each tied to its probability measure pi.

A basis knows how to
  * evaluate its first ``m`` functions at points theta,
  * sample theta ~ pi (for Monte Carlo gradient estimates),
  * provide a quadrature rule for pi (for exact gradients / reference coefficients),
  * report Q_m = sup_theta sum_{i<m} B_i(theta)^2 (Theorem 3.2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Basis(ABC):
    @abstractmethod
    def evaluate(self, theta: np.ndarray, m: int) -> np.ndarray:
        """Return B_0..B_{m-1} at theta, with shape ``theta.shape[:-1] + (m, theta.shape[-1])``."""

    @abstractmethod
    def sample(self, rng: np.random.Generator, size) -> np.ndarray:
        """Draw theta ~ pi."""

    @abstractmethod
    def quadrature(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Nodes and weights (summing to 1) integrating against pi."""

    @abstractmethod
    def Q(self, m: int) -> float:
        """sup_theta sum_{i<m} B_i(theta)^2."""

    def coefficients(self, fn, m: int, n_quad: int = 2**15) -> np.ndarray:
        """Project ``fn(theta) -> (q, n)`` (or ``(n,)``) onto the first m basis functions."""
        nodes, weights = self.quadrature(n_quad)
        values = np.atleast_2d(fn(nodes))
        return (values * weights) @ self.evaluate(nodes, m).T


class TrigBasis(Basis):
    """Trigonometric basis, orthonormal for theta ~ U[-pi, pi].

    B_0 = 1, B_{2j-1} = sqrt(2) cos(j theta), B_{2j} = sqrt(2) sin(j theta).
    """

    def evaluate(self, theta, m):
        theta = np.asarray(theta, dtype=float)
        out = np.empty(theta.shape[:-1] + (m, theta.shape[-1]))
        out[..., 0, :] = 1.0
        for i in range(1, m):
            j = (i + 1) // 2
            trig = np.cos if i % 2 == 1 else np.sin
            out[..., i, :] = np.sqrt(2.0) * trig(j * theta)
        return out

    def sample(self, rng, size):
        return rng.uniform(-np.pi, np.pi, size=size)

    def quadrature(self, n):
        # The trapezoidal rule is exact for trigonometric polynomials of degree < n.
        return -np.pi + 2 * np.pi * np.arange(n) / n, np.full(n, 1.0 / n)

    def Q(self, m):
        # Attained at theta = 0: every cos^2 + sin^2 pair contributes 2, a lone cos contributes 2.
        return float(m + 1 if m % 2 == 0 else m)


class LegendreBasis(Basis):
    """Normalised Legendre polynomials, orthonormal for theta ~ U[-1, 1]."""

    def evaluate(self, theta, m):
        theta = np.asarray(theta, dtype=float)
        out = np.empty(theta.shape[:-1] + (m, theta.shape[-1]))
        p_prev, p = np.ones_like(theta), theta
        for i in range(m):
            if i == 0:
                cur = p_prev
            elif i == 1:
                cur = p
            else:
                p_prev, p = p, ((2 * i - 1) * theta * p - (i - 1) * p_prev) / i
                cur = p
            out[..., i, :] = np.sqrt(2 * i + 1) * cur
        return out

    def sample(self, rng, size):
        return rng.uniform(-1.0, 1.0, size=size)

    def quadrature(self, n):
        nodes, weights = np.polynomial.legendre.leggauss(n)
        return nodes, weights / 2.0

    def Q(self, m):
        # Attained at theta = +-1 where |P_i| = 1.
        return float(m * m)


class TensorLegendreBasis(Basis):
    """Products of normalised Legendre polynomials, orthonormal for theta ~ U[-1, 1]^d.

    Multi-indices are ordered by total degree, so the first m functions span all
    polynomials of total degree <= p whenever m = binom(p + d, d). Points theta have
    shape ``(..., d, n)``: the coordinate axis sits just before the sample axis.
    """

    def __init__(self, d: int, max_degree: int = 40):
        self.d = d
        by_degree = [
            idx
            for total in range(max_degree + 1)
            for idx in sorted(_compositions(total, d), reverse=True)
        ]
        self.index = np.array(by_degree)
        self._legendre = LegendreBasis()

    def evaluate(self, theta, m):
        theta = np.asarray(theta, dtype=float)
        idx = self.index[:m]
        P = self._legendre.evaluate(theta, int(idx.max()) + 1)  # (..., d, deg + 1, n)
        out = np.ones(theta.shape[:-2] + (m, theta.shape[-1]))
        for j in range(self.d):
            out *= P[..., j, idx[:, j], :]
        return out

    def sample(self, rng, size):
        size = tuple(np.atleast_1d(size))
        return rng.uniform(-1.0, 1.0, size=size[:-1] + (self.d, size[-1]))

    def quadrature(self, n):
        """Tensor Gauss-Legendre rule with about n points in total."""
        k = max(2, int(round(n ** (1 / self.d))))
        nodes, weights = self._legendre.quadrature(k)
        grid = np.meshgrid(*[nodes] * self.d, indexing="ij")
        w = np.ones_like(grid[0])
        for wj in np.meshgrid(*[weights] * self.d, indexing="ij"):
            w = w * wj
        return np.stack([g.ravel() for g in grid]), w.ravel()

    def Q(self, m):
        # Attained at theta = (1, ..., 1), where every factor equals sqrt(2 i + 1).
        return float(np.prod(2 * self.index[:m] + 1, axis=1).sum())


def _compositions(total, parts):
    if parts == 1:
        return [(total,)]
    return [(i,) + rest for i in range(total + 1) for rest in _compositions(total - i, parts - 1)]
