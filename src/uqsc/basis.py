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
