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


def test_tensor_legendre_orthonormal_and_Q():
    from uqsc.basis import TensorLegendreBasis

    basis, m = TensorLegendreBasis(2), 10
    nodes, w = basis.quadrature(20**2)
    B = basis.evaluate(nodes, m)
    np.testing.assert_allclose((B * w) @ B.T, np.eye(m), atol=1e-10)
    corner = np.ones((2, 1))
    assert np.sum(basis.evaluate(corner, m) ** 2) == pytest.approx(basis.Q(m))
    assert basis.sample(np.random.default_rng(0), (3, 7)).shape == (3, 2, 7)


def test_markowitz_gradient_and_optimum():
    from uqsc.problems import MarkowitzProblem

    p = MarkowitzProblem()
    theta = np.random.default_rng(0).uniform(-1, 1, (2, 5))
    x = np.random.default_rng(1).normal(size=(p.q, 5))
    h, e = 1e-6, np.eye(p.q)[:, :, None]
    fd = np.stack([(p.value(x + h * e[i], theta) - p.value(x - h * e[i], theta)) / (2 * h) for i in range(p.q)])
    np.testing.assert_allclose(p.mean_grad(x, theta), fd, atol=1e-6)
    np.testing.assert_allclose(p.mean_grad(p.optimum(theta), theta), 0, atol=1e-12)
    assert 0 < p.mu < p.L


def test_baselines_run():
    from uqsc.baselines import adaptive_learn_solution_path, learn_solution_path, naive_monte_carlo
    from uqsc.basis import TensorLegendreBasis
    from uqsc.problems import MarkowitzProblem

    p, b = MarkowitzProblem(), TensorLegendreBasis(2)
    ref = b.coefficients(p.optimum, 6, 20**2)
    lsp = learn_solution_path(p, b, 6, 200, reference=ref, seed=0)
    alsp = adaptive_learn_solution_path(p, b, 1, 6, 200, check_every=50, tol=1.0, reference=ref, seed=0)
    assert lsp.errors[0, -1] < lsp.errors[0, 0]
    assert alsp.levels[-1] > 1
    u = naive_monte_carlo(p, b, 6, 2000, 400, np.random.default_rng(0))
    assert np.sum((u - ref) ** 2) < 1e-2


def test_adaptive_uq_descent_grows_and_converges():
    from uqsc.algorithms import adaptive_uq_descent
    from uqsc.basis import TensorLegendreBasis
    from uqsc.problems import MarkowitzProblem

    p, b = MarkowitzProblem(), TensorLegendreBasis(2)
    ref = b.coefficients(p.optimum, 6, 20**2)
    est = ExactGradient(p, b, 20**2)
    alpha, beta = agd_parameters(p.mu, 1 / p.L)
    gd = adaptive_uq_descent(p, b, est, 1, 6, 300, step_size=1 / (p.mu + p.L), check_every=5, tol=1e-2,
                             reference=ref)
    agd = adaptive_uq_descent(p, b, est, 1, 6, 300, alpha=alpha, beta=beta, check_every=5, tol=1e-2, reference=ref)
    for res in (gd, agd):
        assert res.levels[-1] > 1 and np.all(np.diff(res.levels) >= 0)
        assert res.errors[0, -1] < res.errors[0, 0]
    with pytest.raises(ValueError):
        adaptive_uq_descent(p, b, est, 1, 6, 10)


def _synthetic_logistic(ridge=0.25, sample_rows=False):
    from uqsc.problems import WeightedLogisticProblem

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (rng.uniform(size=60) < 0.3).astype(float)
    return WeightedLogisticProblem(ridge=ridge, sample_rows=sample_rows, data=(X, y))


def test_weighted_logistic_gradient_optimum_and_constants():
    p = _synthetic_logistic()
    theta = np.linspace(-1, 1, 7)
    x = np.random.default_rng(1).normal(size=(p.q, 7))
    h, e = 1e-6, np.eye(p.q)[:, :, None]
    fd = np.stack([(p.value(x + h * e[i], theta) - p.value(x - h * e[i], theta)) / (2 * h) for i in range(p.q)])
    np.testing.assert_allclose(p.mean_grad(x, theta), fd, atol=1e-6)
    np.testing.assert_allclose(p.mean_grad(p.optimum(theta), theta), 0, atol=1e-12)
    np.testing.assert_allclose(p.mean_grad(x, theta, chunk=60), p.mean_grad(x, theta))  # chunked path
    assert p.mu == 0.25 and p.L > p.mu


def test_weighted_logistic_row_sampling_is_unbiased():
    p = _synthetic_logistic(sample_rows=True)
    theta = np.full(200_000, 0.3)
    x = np.tile(np.random.default_rng(2).normal(size=(p.q, 1)), (1, theta.size))
    g = p.grad(x, theta, p.sample_v(np.random.default_rng(3), theta.shape))
    np.testing.assert_allclose(g.mean(axis=1), p.mean_grad(x[:, :1], theta[:1])[:, 0], atol=0.02)
