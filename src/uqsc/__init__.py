"""Uncertainty quantification for gradient and accelerated gradient descent (McMeel & Parpas)."""

from .algorithms import (
    Result,
    adaptive_uq_descent,
    agd_parameters,
    constant_schedule,
    gd_step_size,
    power_schedule,
    sa_step_size,
    sqrt_schedule,
    stochastic_approximation,
    uq_accelerated_gradient_descent,
    uq_gradient_descent,
)
from .baselines import adaptive_learn_solution_path, learn_solution_path, naive_monte_carlo
from .basis import Basis, LegendreBasis, TensorLegendreBasis, TrigBasis
from .gradients import ExactGradient, MonteCarloGradient
from .problems import (
    MarkowitzProblem,
    NoisyQuadraticProblem,
    Problem,
    QuadraticProblem,
    WeightedLogisticProblem,
    crepey_target,
)

__all__ = [
    "Basis",
    "ExactGradient",
    "LegendreBasis",
    "MarkowitzProblem",
    "MonteCarloGradient",
    "NoisyQuadraticProblem",
    "Problem",
    "QuadraticProblem",
    "Result",
    "TensorLegendreBasis",
    "TrigBasis",
    "WeightedLogisticProblem",
    "adaptive_learn_solution_path",
    "adaptive_uq_descent",
    "agd_parameters",
    "constant_schedule",
    "crepey_target",
    "gd_step_size",
    "learn_solution_path",
    "naive_monte_carlo",
    "power_schedule",
    "sa_step_size",
    "sqrt_schedule",
    "stochastic_approximation",
    "uq_accelerated_gradient_descent",
    "uq_gradient_descent",
]
