"""Reproduce the experiments of Section 6.

    uqsc-experiments all
    uqsc-experiments fig1a fig3 --trials 50 --out results

Every figure uses the quadratic (6.2) (or its noisy variant (6.3)) with target x* from
(6.1), the trigonometric basis on theta ~ U[-pi, pi], and plots the mean over trials of
||u^k - P_{m_k}(u*)||^2, which excludes the algorithm-independent truncation error.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .algorithms import (  # noqa: E402
    agd_parameters,
    constant_schedule,
    gd_step_size,
    power_schedule,
    sqrt_schedule,
    stochastic_approximation,
    uq_accelerated_gradient_descent,
    uq_gradient_descent,
)
from .basis import TrigBasis  # noqa: E402
from .gradients import ExactGradient, MonteCarloGradient  # noqa: E402
from .problems import NoisyQuadraticProblem, QuadraticProblem  # noqa: E402

# Embedding matplotlib's bundled cmr10 font makes fontTools warn about its old timestamps.
logging.getLogger("fontTools").setLevel(logging.ERROR)

N_QUAD = 2**14
COLORS = {"GD": "#2a78d6", "AGD": "#eb6834", "SGD": "#1baf7a"}


def _reference(problem, basis, m):
    return basis.coefficients(problem.optimum, m, N_QUAD)


# Half-width figures for a two-column subfigure layout, in Computer Modern to match LaTeX.
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
}


def _draw(curves, colors, figsize, lw, title=None):
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for i, (label, y) in enumerate(curves.items()):
        ax.semilogy(np.arange(1, len(y) + 1), y, lw=lw, label=label, color=colors.get(label, f"C{i}"))
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$\|u^k - P_{m_k}(u^*)\|_2^2$")
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    ax.grid(True, which="major", color="0.88", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if len(curves) > 1:
        ax.legend(frameon=False)
    return fig


def _plot(curves: dict[str, np.ndarray], title: str, path: Path, colors=None, paper=False):
    colors = colors or {}
    fig = _draw(curves, colors, figsize=(6.4, 4.2), lw=1.8, title=title)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    np.savez(path.with_suffix(".npz"), **curves)
    print(f"  saved {path.with_suffix('.png')}")

    if paper:
        with plt.rc_context(PAPER_STYLE):
            fig = _draw(curves, colors, figsize=(3.2, 2.4), lw=1.2)
            fig.savefig(path.with_suffix(".pdf"))
            plt.close(fig)
        print(f"  saved {path.with_suffix('.pdf')}")


def fig1a(args):
    """Exact gradients: GD with gamma = 2/(mu+L); AGD with alpha = 1/L."""
    problem, basis = QuadraticProblem(mu=args.mu, L=args.L), TrigBasis()
    schedule, K = sqrt_schedule(), 300
    ref = _reference(problem, basis, schedule(K))
    est = ExactGradient(problem, basis, N_QUAD)
    common = dict(q=problem.q, reference=ref)

    gd = uq_gradient_descent(est, schedule, K, 2 / (problem.mu + problem.L), **common)
    alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
    agd = uq_accelerated_gradient_descent(est, schedule, K, alpha, beta, **common)
    _plot({"GD": gd.mean_error, "AGD": agd.mean_error}, "Fig. 1a: exact gradients", args.out / "fig1a", COLORS, paper=args.paper)


def fig1b(args):
    """Monte Carlo gradients (M = 500): GD at the maximal step of Theorem 4.3, AGD, and SA."""
    problem, basis = QuadraticProblem(mu=args.mu, L=args.L), TrigBasis()
    schedule, K = sqrt_schedule(), 300
    ref = _reference(problem, basis, schedule(K))
    est = MonteCarloGradient(problem, basis, n_samples=500)
    common = dict(q=problem.q, reference=ref, n_trials=args.trials)

    gd = uq_gradient_descent(est, schedule, K, gd_step_size(problem, basis, est), seed=args.seed, **common)
    alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
    agd = uq_accelerated_gradient_descent(est, schedule, K, alpha, beta, seed=args.seed + 1, **common)
    sa = stochastic_approximation(est, schedule, K, gamma0=1e-2, seed=args.seed + 2, **common)
    curves = {"GD": gd.mean_error, "AGD": agd.mean_error, "SGD": sa.mean_error}
    _plot(curves, f"Fig. 1b: Monte Carlo gradients (M=500, {args.trials} trials)", args.out / "fig1b", COLORS, paper=args.paper)


def fig2(args):
    """Noisy objective (6.3) with v ~ U[-1, 1]; GD with the same step sizes as Fig. 1b."""
    problem, basis = NoisyQuadraticProblem(mu=args.mu, L=args.L), TrigBasis()
    schedule, K = sqrt_schedule(), 300
    ref = _reference(problem, basis, schedule(K))
    est = MonteCarloGradient(problem, basis, n_samples=500)

    gd = uq_gradient_descent(
        est, schedule, K, gd_step_size(problem, basis, est),
        q=problem.q, reference=ref, n_trials=args.trials, seed=args.seed,
    )
    _plot({"GD": gd.mean_error}, f"Fig. 2: GD with uncertainty in v ({args.trials} trials)", args.out / "fig2", COLORS, paper=args.paper)


def fig3(args):
    """UQ (growing m_k) versus a fixed, large level, both with M = 250 and maximal GD steps."""
    problem, basis = QuadraticProblem(mu=1.0, L=200.0), TrigBasis()
    schedule, K = power_schedule(args.power), 600
    m_fixed = args.fixed_m or schedule(K)
    ref = _reference(problem, basis, max(m_fixed, schedule(K)))
    est = MonteCarloGradient(problem, basis, n_samples=250)
    trunc = np.sum(ref[:, m_fixed:] ** 2) + np.sum(basis.coefficients(problem.optimum, 4096, N_QUAD)[:, ref.shape[1]:] ** 2)
    print(f"  m_k = floor((k+10)^{args.power}) + 2 reaches m = {schedule(K)}; fixed level m = {m_fixed}, ||R_m||^2 = {trunc:.2e} (both components), "
          f"C_G = {1 + 2 * basis.Q(m_fixed) / 250:.3f}")

    curves = {}
    for label, sched in [("Full UQ method", schedule), ("Fixed amount of basis functions", constant_schedule(m_fixed))]:
        t0 = time.perf_counter()
        res = uq_gradient_descent(
            est, sched, K, gd_step_size(problem, basis, est),
            q=problem.q, reference=ref, n_trials=args.trials, seed=args.seed,
        )
        print(f"  {label}: {(time.perf_counter() - t0):.2f}s for {args.trials} trials")
        curves[label] = res.mean_error
    colors = {"Full UQ method": COLORS["GD"], "Fixed amount of basis functions": COLORS["AGD"]}
    _plot(curves, f"Fig. 3: fixed vs growing basis ({args.trials} trials)", args.out / "fig3", colors, paper=args.paper)


FIGURES = {"fig1a": fig1a, "fig1b": fig1b, "fig2": fig2, "fig3": fig3}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figures", nargs="*", default=["all"], choices=[*FIGURES, "all"])
    parser.add_argument("--trials", type=int, default=200, help="Monte Carlo repetitions averaged per curve")
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--L", type=float, default=200.0)
    parser.add_argument("--power", type=float, default=0.7, help="fig3 schedule m_k = floor((k+10)^power) + 2")
    parser.add_argument("--fixed-m", type=int, default=None, help="fixed level for fig3 (default: max m_k reached)")
    parser.add_argument("--paper", action="store_true", help="also write half-width Computer Modern PDFs for LaTeX")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    names = list(FIGURES) if "all" in args.figures else args.figures
    for name in names:
        print(f"{name}:")
        FIGURES[name](args)


if __name__ == "__main__":
    main()
