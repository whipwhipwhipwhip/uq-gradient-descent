"""Uncertainty quantification for gradient and accelerated gradient descent (McMeel & Parpas)."""

from .algorithms import (
    Result,
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
from .basis import Basis, LegendreBasis, TrigBasis
from .gradients import ExactGradient, MonteCarloGradient
from .problems import NoisyQuadraticProblem, Problem, QuadraticProblem, crepey_target

__all__ = [
    "Basis",
    "ExactGradient",
    "LegendreBasis",
    "MonteCarloGradient",
    "NoisyQuadraticProblem",
    "Problem",
    "QuadraticProblem",
    "Result",
    "TrigBasis",
    "agd_parameters",
    "constant_schedule",
    "crepey_target",
    "gd_step_size",
    "power_schedule",
    "sa_step_size",
    "sqrt_schedule",
    "stochastic_approximation",
    "uq_accelerated_gradient_descent",
    "uq_gradient_descent",
]
