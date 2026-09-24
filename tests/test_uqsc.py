import numpy as np
import pytest

from uqsc import (
    ExactGradient,
    LegendreBasis,
    MonteCarloGradient,
    QuadraticProblem,
    TrigBasis,
    agd_parameters,
    constant_schedule,
    sqrt_schedule,
    uq_accelerated_gradient_descent,
    uq_gradient_descent,
)


@pytest.mark.parametrize("basis", [TrigBasis(), LegendreBasis()])
def test_basis_orthonormal_and_Q(basis):
    m = 9
    nodes, w = basis.quadrature(256)
    B = basis.evaluate(nodes, m)
    np.testing.assert_allclose((B * w) @ B.T, np.eye(m), atol=1e-10)
    grid = np.linspace(nodes.min(), nodes.max(), 1001)
    grid = np.concatenate([grid, [-1.0, 0.0, 1.0]]) if isinstance(basis, LegendreBasis) else grid
    assert np.max(np.sum(basis.evaluate(grid, m) ** 2, axis=0)) == pytest.approx(basis.Q(m), rel=1e-6)


def test_monte_carlo_gradient_is_unbiased():
    problem, basis = QuadraticProblem(), TrigBasis()
    u = np.random.default_rng(1).normal(size=(1, 2, 5))
    exact = ExactGradient(problem, basis).estimate(u, 5, 1, None)
    mc = MonteCarloGradient(problem, basis, n_samples=200_000).estimate(u, 5, 1, np.random.default_rng(0))
    np.testing.assert_allclose(mc, exact, atol=0.05 * problem.L)


@pytest.mark.parametrize("algorithm", ["gd", "agd"])
def test_exact_gradient_converges_to_truncated_optimum(algorithm):
    problem, basis = QuadraticProblem(mu=1.0, L=20.0), TrigBasis()
    m = 7
    ref = basis.coefficients(problem.optimum, m, 2**14)
    est = ExactGradient(problem, basis, 2**14)
    if algorithm == "gd":
        res = uq_gradient_descent(est, constant_schedule(m), 400, 2 / (problem.mu + problem.L), q=2, reference=ref)
    else:
        alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
        res = uq_accelerated_gradient_descent(est, constant_schedule(m), 400, alpha, beta, q=2, reference=ref)
    assert res.errors[0, -1] < 1e-12
    np.testing.assert_allclose(res.u[0], ref, atol=1e-6)


def test_growing_levels_are_respected():
    problem, basis = QuadraticProblem(), TrigBasis()
    res = uq_gradient_descent(ExactGradient(problem, basis), sqrt_schedule(), 50, 0.005, q=2)
    assert list(res.levels[:3]) == [5, 5, 5]
    assert res.u.shape == (1, 2, res.levels[-1])
