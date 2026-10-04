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
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .algorithms import (  # noqa: E402
    adaptive_uq_descent,
    agd_parameters,
    constant_schedule,
    gd_step_size,
    power_schedule,
    sqrt_schedule,
    stochastic_approximation,
    uq_accelerated_gradient_descent,
    uq_gradient_descent,
)
from .baselines import adaptive_learn_solution_path, learn_solution_path, naive_monte_carlo  # noqa: E402
from .basis import TensorLegendreBasis, TrigBasis  # noqa: E402
from .gradients import ExactGradient, MonteCarloGradient  # noqa: E402
from .problems import MarkowitzProblem, NoisyQuadraticProblem, QuadraticProblem  # noqa: E402

# Embedding matplotlib's bundled cmr10 font makes fontTools warn about its old timestamps.
logging.getLogger("fontTools").setLevel(logging.ERROR)

N_QUAD = 2**14
COLORS = {"GD": "#2a78d6", "AGD": "#eb6834", "SGD": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


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


def _style_axes(ax):
    ax.grid(True, which="major", color="0.88", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _draw(curves, colors, figsize, lw, title=None, xlabel="Iteration $k$",
          ylabel=r"$\|u^k - P_{m_k}(u^*)\|_2^2$", logx=False, styles=None):
    """Line plot on a log y-axis.

    Values of ``curves`` are y arrays, (x, y) pairs, or (x, y, lo, hi) tuples, in which case
    the band between lo and hi (the spread across trials) is shaded.
    """
    styles = styles or {}
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for i, (label, c) in enumerate(curves.items()):
        x, y, *band = c if isinstance(c, tuple) else (np.arange(1, len(c) + 1), c)
        color = colors.get(label, f"C{i}")
        if band:
            ax.fill_between(x, band[0], band[1], color=color, alpha=0.15, lw=0)
        ax.plot(x, y, lw=lw, label=label, color=color, **styles.get(label, {}))
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    _style_axes(ax)
    if len(curves) > 1:
        ax.legend(frameon=False)
    return fig


def _save(build, path: Path, paper: bool, data: dict):
    """Write ``build(paper=False)`` as PNG, ``build(paper=True)`` as PDF if requested, and the data."""
    fig = build(paper=False)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    arrays = {}
    for key, value in data.items():
        name = "".join(ch if ch.isalnum() else "_" for ch in key)
        if isinstance(value, tuple):
            for suffix, arr in zip(("x", "y", "lo", "hi"), value):
                arrays[f"{name}_{suffix}"] = arr
        else:
            arrays[name] = value
    np.savez(path.with_suffix(".npz"), **arrays)
    print(f"  saved {path.with_suffix('.png')}")
    if paper:
        with plt.rc_context(PAPER_STYLE):
            fig = build(paper=True)
            fig.savefig(path.with_suffix(".pdf"))
            plt.close(fig)
        print(f"  saved {path.with_suffix('.pdf')}")


def _plot(curves: dict, title: str, path: Path, colors=None, paper=False, paper_size=(3.2, 2.4), **opts):
    colors = colors or {}

    def build(paper):
        if paper:
            return _draw(curves, colors, figsize=paper_size, lw=1.2, **opts)
        return _draw(curves, colors, figsize=(6.4, 4.2), lw=1.8, title=title, **opts)

    _save(build, path, paper, curves)


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
    noisy = NoisyQuadraticProblem(mu=args.mu, L=args.L)
    noisy_est = MonteCarloGradient(noisy, basis, n_samples=500)
    gd_v = uq_gradient_descent(noisy_est, schedule, K, gd_step_size(noisy, basis, noisy_est), seed=args.seed + 3,
                               q=noisy.q, reference=ref, n_trials=args.trials)
    curves = {"GD": gd.mean_error, "AGD": agd.mean_error, "SGD": sa.mean_error, "GD, noise in $v$": gd_v.mean_error}
    colors = {**COLORS, "GD, noise in $v$": COLORS["GD"]}
    styles = {"GD, noise in $v$": dict(ls="--")}
    _plot(curves, f"Fig. 1b: Monte Carlo gradients (M=500, {args.trials} trials)", args.out / "fig1b", colors,
          paper=args.paper, styles=styles)


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


# ---------------------------------------------------------------------------
# Additional experiments: Markowitz problem of Example 1.1 with theta in [-1, 1]^2
# ---------------------------------------------------------------------------

MK_DEGREE = 6  # final total degree of the tensor Legendre basis, m = 28 functions
MK_REF_DEGREE = 20  # reference coefficients up to total degree 20 (231 functions)
MK_QUAD = 60**2
MK_NOISE = 0.05  # v ~ U[-0.05, 0.05]^n on the returns
MK_STALL_WINDOW = 5  # iterations between stall checks for the adaptive UQ methods (fallback)
# Stall-rule settings per adaptive variant, keyed by (method, growth unit). Each was chosen by a
# sweep as the setting reaching 1e-6 in the fewest gradient evaluations (ties at 1e-8):
# ALSP by (steps between checks, threshold); our methods by iterations between checks.
ALSP_CHECK_EVERY = 1000
ALSP_TOL = 1e-2
STALL = {
    ("ALSP", "one degree"): (5000, 1e-2),
    ("UQ-GD", "one function"): 5,
    ("UQ-AGD", "one function"): 2,
    ("UQ-GD", "one degree"): 5,
    ("UQ-AGD", "one degree"): 5,
}
BAND = (10, 90)  # percentiles of the shaded spread across trials


def _n_functions(degree, d=2):
    return int(np.prod(range(degree + 1, degree + d + 1)) // np.prod(range(1, d + 1)))


def _markowitz(noise=0.0):
    problem, basis = MarkowitzProblem(return_noise=noise), TensorLegendreBasis(2)
    ref = basis.coefficients(problem.optimum, _n_functions(MK_REF_DEGREE), MK_QUAD)
    total = np.sum(ref**2)
    tail = lambda m: total - np.sum(ref[:, :m] ** 2)  # noqa: E731  ||R_m||^2
    return problem, basis, ref, tail


def _degree_schedule(every=25, start=1, final=MK_DEGREE):
    """Add all polynomials of the next total degree every ``every`` iterations."""
    return lambda k: _n_functions(min(final, start + (k - 1) // every))


def _proportional_samples(basis, factor=2):
    """M_k = factor * Q_{m_k}, so C_G = 1 + 2 / factor stays constant (Theorem 4.4)."""
    return lambda k, m: int(np.ceil(factor * basis.Q(m)))


def markowitz(args):
    """UQ-GD and UQ-AGD on Example 1.1 with noisy returns and a growing 2-D Legendre basis."""
    problem, basis, ref, _ = _markowitz(MK_NOISE)
    est = MonteCarloGradient(problem, basis, n_samples=_proportional_samples(basis))
    n_trials, K = max(5, args.trials // 10), 200
    common = dict(q=problem.q, reference=ref, n_trials=n_trials)
    print(f"  kappa = {problem.kappa:.1f}, V = {problem.V:.2e}, {n_trials} trials")

    gd = uq_gradient_descent(est, _degree_schedule(), K, gd_step_size(problem, basis, est), seed=args.seed, **common)
    alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
    agd = uq_accelerated_gradient_descent(est, _degree_schedule(), K, alpha, beta, seed=args.seed + 1, **common)
    curves = {"GD": gd.mean_error, "AGD": agd.mean_error}
    _plot(curves, f"Markowitz, θ ∈ [-1,1]², noisy returns ({n_trials} trials)", args.out / "markowitz",
          COLORS, paper=args.paper)


def lemma31(args):
    """Distance between the level-m optimum and the projection of u*, against Lemma 3.1."""
    problem, basis, ref, tail = _markowitz()
    est = ExactGradient(problem, basis, MK_QUAD)
    alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
    ms, gaps = [], []
    for degree in range(0, 9):
        m = _n_functions(degree)
        res = uq_accelerated_gradient_descent(est, constant_schedule(m), 3000, alpha, beta, q=problem.q, reference=ref)
        ms.append(m)
        gaps.append(res.errors[0, -1])
    ms = np.array(ms, dtype=float)
    R = np.array([tail(int(m)) for m in ms])
    curves = {
        r"$\|P_m(u^*) - u^*_m\|_2^2$": (ms, np.array(gaps)),
        r"$\|R_m\|_2^2$": (ms, R),
        r"$\frac{(\kappa-1)^2}{4\kappa} \|R_m\|_2^2$ (Lemma 3.1)": (ms, (problem.kappa - 1) ** 2 / (4 * problem.kappa) * R),
    }
    colors = dict(zip(curves, SERIES))
    styles = dict(zip(curves, [dict(marker="o", ms=4), dict(marker="s", ms=4, ls="--"), dict(ls=":")]))
    for label, (m, y) in curves.items():
        print(f"  {label}: " + ", ".join(f"m={int(a)}: {b:.1e}" for a, b in zip(m, y)))
    _plot(curves, "Truncated optimum vs. projection (Markowitz)", args.out / "lemma31", colors,
          paper=args.paper, xlabel="Number of basis functions $m$", ylabel="Squared distance", styles=styles)


def markowitz_uq(args):
    """The UQ output itself: distribution of the optimal weights, learned vs. ground truth."""
    problem, basis, ref, _ = _markowitz(MK_NOISE)
    est = MonteCarloGradient(problem, basis, n_samples=_proportional_samples(basis))
    alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
    res = uq_accelerated_gradient_descent(est, _degree_schedule(), 600, alpha, beta, q=problem.q, seed=args.seed)
    u = res.u[0]

    theta = basis.sample(np.random.default_rng(args.seed + 99), (20000,))
    learned = u @ basis.evaluate(theta, u.shape[1])
    truth = problem.optimum(theta)

    def stats(x):
        q05, q95 = np.quantile(x, [0.05, 0.95], axis=1)
        return np.stack([x.mean(1), x.std(1), q05, q95, (x < 0).mean(1)], axis=1)

    s_true, s_learned = stats(truth), stats(learned)
    _write_uq_table(s_true, s_learned, args.out / "markowitz_uq_table")
    uncertain = np.flatnonzero((s_true[:, 4] > 0.05) & (s_true[:, 4] < 0.95))
    assets = uncertain[np.argsort(-s_true[uncertain, 1])][:3]
    print(f"  max |mean error| = {np.abs(s_true[:, 0] - s_learned[:, 0]).max():.1e}, "
          f"max |P(w<0) error| = {np.abs(s_true[:, 4] - s_learned[:, 4]).max():.3f}")

    def build(paper):
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.1) if paper else (10, 3.4), layout="constrained")
        for ax, i in zip(axes, assets):
            bins = np.linspace(min(truth[i].min(), learned[i].min()), max(truth[i].max(), learned[i].max()), 50)
            ax.hist(truth[i], bins=bins, density=True, color="0.82", label="Ground truth")
            ax.hist(learned[i], bins=bins, density=True, histtype="step", lw=1.2 if paper else 1.8,
                    color=SERIES[0], label="UQ-AGD")
            ax.axvline(0, color="0.4", lw=0.6)
            ax.set_xlabel(f"$w_{{{i + 1}}}^*(\\theta)$")
            ax.set_yticks([])
            _style_axes(ax)
            ax.grid(False)
            ax.set_title(f"$P(w_{{{i + 1}}}^* < 0) = {s_true[i, 4]:.2f}$ (learned {s_learned[i, 4]:.2f})",
                         fontsize=8 if paper else 10)
        axes[0].legend(frameon=False, loc="upper left")
        if not paper:
            fig.suptitle("Distribution of optimal weights under covariance uncertainty", x=0.01, ha="left")
        return fig

    _save(build, args.out / "markowitz_uq", args.paper, {"truth": truth[assets], "learned": learned[assets]})


def _write_uq_table(s_true, s_learned, path):
    cols = ["mean", "sd", "5%", "95%", "P(w<0)"]
    lines = ["| asset | " + " | ".join(f"{c} (true / learned)" for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    tex = [r"\begin{tabular}{c" + "cc" * len(cols) + "}", r"\toprule",
           "Asset & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{c.replace('%', chr(92) + '%')}}}" for c in cols)
           + r" \\", " & " + " & ".join(["True & Learned"] * len(cols)) + r" \\", r"\midrule"]
    for i, (t, l) in enumerate(zip(s_true, s_learned)):
        lines.append(f"| {i + 1} | " + " | ".join(f"{a:.3f} / {b:.3f}" for a, b in zip(t, l)) + " |")
        tex.append(f"{i + 1} & " + " & ".join(f"{a:.3f} & {b:.3f}" for a, b in zip(t, l)) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    path.with_suffix(".tex").write_text("\n".join(tex) + "\n")
    print(f"  saved {path.with_suffix('.md')} and .tex")


def _log_subsample(x, y, n=400):
    idx = np.unique(np.geomspace(1, len(x), n).astype(int)) - 1
    return x[idx], y[idx]


def cost(args):
    """Full L2_pi error against gradient evaluations: UQ methods vs. Dong et al. vs. naive MC (V = 0)."""
    problem, basis, ref, tail = _markowitz()
    m_final = _n_functions(MK_DEGREE)
    n_trials = max(5, args.trials // 20)
    common = dict(reference=ref, n_trials=n_trials)
    curves, t0 = {}, time.perf_counter()

    def full(res):
        return res.mean_error + np.array([tail(m) for m in res.levels])

    est = MonteCarloGradient(problem, basis, n_samples=_proportional_samples(basis))
    alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
    for label, res in [
        ("UQ-GD", uq_gradient_descent(est, _degree_schedule(), 2000, gd_step_size(problem, basis, est),
                                      q=problem.q, seed=args.seed, **common)),
        ("UQ-AGD", uq_accelerated_gradient_descent(est, _degree_schedule(), 800, alpha, beta,
                                                   q=problem.q, seed=args.seed + 1, **common)),
    ]:
        evals = np.cumsum([est.n_samples(k, m) for k, m in enumerate(res.levels, start=1)])
        curves[label] = (evals.astype(float), full(res))
    print(f"  UQ methods: {time.perf_counter() - t0:.0f}s")

    n_sgd = 10**6
    t0 = time.perf_counter()
    lsp = learn_solution_path(problem, basis, m_final, n_sgd, seed=args.seed + 2, **common)
    curves[f"LSP (m={m_final})"] = _log_subsample(np.arange(1.0, n_sgd + 1), full(lsp))
    alsp = adaptive_learn_solution_path(problem, basis, 3, m_final, n_sgd, check_every=5000, tol=1e-3,
                                        seed=args.seed + 3, **common)
    curves["ALSP"] = _log_subsample(np.arange(1.0, n_sgd + 1), full(alsp))
    print(f"  Dong et al. baselines: {time.perf_counter() - t0:.0f}s (ALSP reached m={alsp.levels[-1]})")

    # Naive MC: per-sample GD solved to squared relative accuracy 1e-8, best truncation in hindsight.
    rho = (problem.kappa - 1) / (problem.kappa + 1)
    solver_iters = int(np.ceil(np.log(1e-4) / np.log(rho)))
    levels = [_n_functions(p) for p in range(0, 9)]
    sizes = np.unique(np.geomspace(30, 30000, 10).astype(int))
    rng = np.random.default_rng(args.seed + 4)
    errs = []
    for N in sizes:
        trial_errs = []
        for _ in range(n_trials):
            u = naive_monte_carlo(problem, basis, levels[-1], int(N), solver_iters, rng)
            trial_errs.append([np.sum((u[:, :m] - ref[:, :m]) ** 2) + tail(m) for m in levels])
        errs.append(np.min(np.mean(trial_errs, axis=0)))
    curves["Naive MC"] = (sizes * float(solver_iters), np.array(errs))

    colors = dict(zip(curves, SERIES))
    styles = {f"LSP (m={m_final})": dict(ls="--"), "ALSP": dict(ls="-."), "Naive MC": dict(ls=":", marker="o", ms=3)}
    for label, (x, y) in curves.items():
        print(f"  {label}: final error {y[-1]:.1e} after {x[-1]:.1e} gradient evaluations")
    _plot(curves, f"Cost to accuracy (Markowitz, V = 0, {n_trials} trials)", args.out / "cost", colors,
          paper=args.paper, logx=True, styles=styles, xlabel=r"Gradient evaluations of $F$",
          ylabel=r"$\|x - x^*\|_\pi^2$")


def _record_points(n, k=400):
    """About k log-spaced iterations in 1..n at which to record errors."""
    return np.unique(np.geomspace(1, n, k).astype(int))


def _stats(res, tail):
    """Mean and BAND percentiles over trials of the full error ||u - u*||^2 at the recorded iterations."""
    full = res.errors + np.array([tail(m) for m in res.recorded_levels])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns after a divergence
        lo, hi = np.nanpercentile(full, BAND, axis=0)
    return full.mean(axis=0), lo, hi


def _sgd_curve(res, tail):
    """(evaluations, mean, lo, hi) for a method using one gradient evaluation per step."""
    x = (res.iterations if res.iterations is not None else np.arange(1, res.errors.shape[1] + 1)).astype(float)
    return (x, *_stats(res, tail))


def _batched_curve(res, est, tail):
    """(evaluations, mean, lo, hi) for a method using est.n_samples(k, m) evaluations per iteration."""
    evals = np.cumsum([est.n_samples(k, m) for k, m in enumerate(res.levels, start=1)]).astype(float)
    x = evals if res.iterations is None else evals[res.iterations - 1]
    return (x, *_stats(res, tail))


def _naive_curve(problem, basis, ref, tail, max_degree, n_trials, rng):
    """Naive MC: per-sample GD to squared relative accuracy 1e-8, best truncation level in hindsight."""
    rho = (problem.kappa - 1) / (problem.kappa + 1)
    solver_iters = int(np.ceil(np.log(1e-4) / np.log(rho)))
    levels = [_n_functions(p) for p in range(0, max_degree + 1)]
    sizes = np.unique(np.geomspace(30, 30000, 10).astype(int))
    mean, lo, hi = [], [], []
    for N in sizes:
        trial_errs = np.array([
            [np.sum((u[:, :m] - ref[:, :m]) ** 2) + tail(m) for m in levels]
            for u in (naive_monte_carlo(problem, basis, levels[-1], int(N), solver_iters, rng) for _ in range(n_trials))
        ])
        best = np.argmin(trial_errs.mean(axis=0))
        mean.append(trial_errs[:, best].mean())
        b = np.percentile(trial_errs[:, best], BAND)
        lo.append(b[0])
        hi.append(b[1])
    return sizes * float(solver_iters), np.array(mean), np.array(lo), np.array(hi)


def _reach(x, y, t):
    hit = np.flatnonzero(y <= t)
    return x[hit[0]] if len(hit) else np.nan


def _reach_table(curves, levels=(1e-4, 1e-6, 1e-8)):
    for label, (x, y, *_) in curves.items():
        reach = {f"{t:.0e}": (f"{_reach(x, y, t):.0e}" if np.isfinite(_reach(x, y, t)) else "-") for t in levels}
        finite = np.isfinite(y)
        print(f"  {label}: final {y[finite][-1]:.1e} after {x[finite][-1]:.1e} evaluations; first reaches {reach}")


def _lsp_curves(problem, basis, tail, common, seed):
    """LSP at total degree 3, 6 and 10 (m = 10, 28, 66)."""
    curves = {}
    for i, (degree, n_sgd) in enumerate([(3, 5 * 10**5), (6, 10**6), (10, 4 * 10**6)]):
        t0 = time.perf_counter()
        m = _n_functions(degree)
        res = learn_solution_path(problem, basis, m, n_sgd, seed=seed + i, record_at=_record_points(n_sgd), **common)
        curves[f"LSP, $m={m}$"] = _sgd_curve(res, tail)
        print(f"  LSP m={m}: {time.perf_counter() - t0:.0f}s")
    return curves


def _alsp(problem, basis, m_max, tail, common, seed, n_sgd=4 * 10**6, grow=None,
          check_every=None, tol=None):
    t0 = time.perf_counter()
    res = adaptive_learn_solution_path(problem, basis, 3, m_max, n_sgd, check_every=check_every or ALSP_CHECK_EVERY,
                                       tol=tol or ALSP_TOL, grow=grow, seed=seed, record_at=_record_points(n_sgd),
                                       **common)
    print(f"  ALSP: {time.perf_counter() - t0:.0f}s (reached m={res.levels[-1]})")
    return _sgd_curve(res, tail)


LSP_STYLES = [dict(ls=(0, (1.5, 1.5))), dict(ls=(0, (4, 1.5))), dict(ls=(0, (8, 2)))]


def comparison(args):
    """Error vs. gradient evaluations when the basis size is not known in advance (V = 0).

    LSP (Dong et al.) at three fixed sizes, ALSP and the UQ methods allowed to grow up to
    total degree 10 (m = 66), and naive per-sample Monte Carlo. Shaded bands are the
    BAND percentiles across trials.
    """
    problem, basis, ref, tail = _markowitz()
    degree_max, n_trials = 10, args.cost_trials
    common = dict(reference=ref, n_trials=n_trials)
    curves, t0 = {}, time.perf_counter()

    est = MonteCarloGradient(problem, basis, n_samples=_proportional_samples(basis))
    schedule = _degree_schedule(final=degree_max)
    alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
    curves["UQ-GD"] = _batched_curve(uq_gradient_descent(est, schedule, 2500, gd_step_size(problem, basis, est),
                                                         q=problem.q, seed=args.seed, **common), est, tail)
    curves["UQ-AGD"] = _batched_curve(uq_accelerated_gradient_descent(est, schedule, 1200, alpha, beta, q=problem.q,
                                                                      seed=args.seed + 1, **common), est, tail)
    print(f"  UQ methods: {time.perf_counter() - t0:.0f}s")
    lsp = _lsp_curves(problem, basis, tail, common, args.seed + 2)
    curves.update(lsp)
    curves["ALSP"] = _alsp(problem, basis, _n_functions(degree_max), tail, common, args.seed + 5)
    curves["Naive MC"] = _naive_curve(problem, basis, ref, tail, degree_max, n_trials,
                                      np.random.default_rng(args.seed + 6))

    # One hue per method; the three LSP sizes share a hue and differ by dash pattern.
    colors = {"UQ-GD": SERIES[0], "UQ-AGD": SERIES[1], "ALSP": SERIES[3], "Naive MC": SERIES[4],
              **{label: SERIES[2] for label in lsp}}
    styles = {**dict(zip(lsp, LSP_STYLES)), "ALSP": dict(ls="-."), "Naive MC": dict(ls=":", marker="o", ms=3)}
    _reach_table(curves)
    _plot(curves, f"Comparison without a known basis size (Markowitz, V = 0, {n_trials} trials)",
          args.out / "comparison", colors, paper=args.paper, paper_size=(6.5, 3.0), logx=True, styles=styles,
          xlabel=r"Gradient evaluations of $F$", ylabel=r"$\|x - x^*\|_\pi^2$")


def schedules(args):
    """Sensitivity to the growth schedule: one extra total degree every 25, 75 or 200 iterations (V = 0)."""
    problem, basis, ref, tail = _markowitz()
    degree_max = 10
    n_trials = max(5, args.trials // 40)
    est = MonteCarloGradient(problem, basis, n_samples=_proportional_samples(basis))
    alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
    every = [25, 75, 200]
    panels = {"UQ-GD": {}, "UQ-AGD": {}}
    for i, e in enumerate(every):
        schedule = _degree_schedule(every=e, final=degree_max)
        label = f"every {e} iterations"
        runs = {
            "UQ-GD": uq_gradient_descent(est, schedule, 1500 + 10 * e, gd_step_size(problem, basis, est),
                                         q=problem.q, reference=ref, n_trials=n_trials, seed=args.seed + i),
            "UQ-AGD": uq_accelerated_gradient_descent(est, schedule, 600 + 10 * e, alpha, beta, q=problem.q,
                                                      reference=ref, n_trials=n_trials, seed=args.seed + 10 + i),
        }
        for method, res in runs.items():
            evals = np.cumsum([est.n_samples(k, m) for k, m in enumerate(res.levels, start=1)]).astype(float)
            panels[method][label] = (evals, res.mean_error + np.array([tail(m) for m in res.levels]))
    for method, curves in panels.items():
        print(f"  {method}:")
        _reach_table(curves)

    colors = dict(zip(panels["UQ-GD"], [SERIES[0], SERIES[1], SERIES[2]]))

    def build(paper):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6) if paper else (11, 4.2), sharey=True, layout="constrained")
        for ax, (method, curves) in zip(axes, panels.items()):
            for label, (x, y) in curves.items():
                ax.plot(x, y, lw=1.2 if paper else 1.8, color=colors[label], label=label)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"Gradient evaluations of $F$")
            ax.set_title(method, loc="left", fontsize=9 if paper else 11)
            _style_axes(ax)
        axes[0].set_ylabel(r"$\|x - x^*\|_\pi^2$")
        axes[0].legend(frameon=False, title="New total degree", title_fontsize=7 if paper else 9)
        if not paper:
            fig.suptitle(f"Sensitivity to the growth schedule (Markowitz, V = 0, {n_trials} trials)", x=0.01, ha="left")
        return fig

    data = {f"{method} {label}": c for method, curves in panels.items() for label, c in curves.items()}
    _save(build, args.out / "schedules", args.paper, data)


def comparison_adaptive(args):
    """As ``comparison``, but both UQ methods grow the basis with ALSP's stall rule.

    Adaptive UQ-GD vs. ALSP isolates mini-batching; adaptive UQ-AGD vs. adaptive UQ-GD
    isolates acceleration, since all three use the same growth heuristic (V = 0).
    """
    problem, basis, ref, tail = _markowitz()
    m_max, n_trials = _n_functions(10), args.cost_trials
    common = dict(reference=ref, n_trials=n_trials)

    lsp = _lsp_curves(problem, basis, tail, common, args.seed + 2)
    curves = dict(lsp)
    curves["ALSP"] = _alsp(problem, basis, m_max, tail, common, args.seed + 5)
    est = MonteCarloGradient(problem, basis, n_samples=_proportional_samples(basis))
    alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
    stall = dict(tol=1e-3, seed=args.seed, **common)
    curves["Adaptive UQ-GD"] = _batched_curve(adaptive_uq_descent(
        problem, basis, est, 3, m_max, 3000, step_size=gd_step_size(problem, basis, est),
        check_every=STALL["UQ-GD", "one function"], **stall), est, tail)
    curves["Adaptive UQ-AGD"] = _batched_curve(adaptive_uq_descent(
        problem, basis, est, 3, m_max, 1500, alpha=alpha, beta=beta,
        check_every=STALL["UQ-AGD", "one function"], **stall), est, tail)

    colors = {"Adaptive UQ-GD": SERIES[0], "Adaptive UQ-AGD": SERIES[1], "ALSP": SERIES[3],
              **{label: SERIES[2] for label in lsp}}
    styles = {**dict(zip(lsp, LSP_STYLES)), "ALSP": dict(ls="-.")}
    _reach_table(curves)
    _plot(curves, f"Same growth heuristic for all adaptive methods (Markowitz, V = 0, {n_trials} trials)",
          args.out / "comparison_adaptive", colors, paper=args.paper, paper_size=(6.5, 3.0), logx=True,
          styles=styles, xlabel=r"Gradient evaluations of $F$", ylabel=r"$\|x - x^*\|_\pi^2$")


def ablation_batch1(args):
    """Can acceleration work at batch size 1? Accelerated SGD with ALSP's stall rule (V = 0).

    Steps are c / (Q_m L) for c in {1, 0.1, 0.01}, with c = 1 the ALSP step, and
    beta = (1 - sqrt(alpha mu)) / (1 + sqrt(alpha mu)). Runs that diverge are stopped and
    marked with a cross. ALSP and batched adaptive UQ-AGD are shown for reference.
    """
    problem, basis, ref, tail = _markowitz()
    m_max, n_trials, n_sgd = _n_functions(10), args.cost_trials, 2 * 10**6
    common = dict(reference=ref, n_trials=n_trials)
    t0 = time.perf_counter()
    curves = {"ALSP (SGD, batch 1)": _alsp(problem, basis, m_max, tail, common, args.seed + 5, n_sgd=n_sgd)}
    batch1 = MonteCarloGradient(problem, basis, n_samples=1)
    diverged = {}
    for i, c in enumerate([1.0, 0.1, 0.01]):
        res = adaptive_uq_descent(problem, basis, batch1, 3, m_max, n_sgd,
                                  alpha=lambda k, m, c=c: c / (basis.Q(m) * problem.L),
                                  check_every=ALSP_CHECK_EVERY, tol=ALSP_TOL, seed=args.seed + 20 + i,
                                  record_at=_record_points(n_sgd), **common)
        label = "Accelerated SGD, batch 1, $\\alpha = 1/(Q_m L)$" if c == 1 else \
            f"Accelerated SGD, batch 1, $\\alpha = {c:g}/(Q_m L)$"
        x, y, lo, hi = _sgd_curve(res, tail)
        ok = np.isfinite(y)
        curves[label] = (x[ok], y[ok], lo[ok], hi[ok])
        diverged[label] = not ok.all()
        status = "diverged" if diverged[label] else "finished"
        print(f"  c = {c:g}: {status} after about {x[ok][-1]:.1e} steps at m = {res.recorded_levels[ok][-1]}")
    est = MonteCarloGradient(problem, basis, n_samples=_proportional_samples(basis))
    alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
    res = adaptive_uq_descent(problem, basis, est, 3, m_max, 1500, alpha=alpha, beta=beta,
                              check_every=STALL["UQ-AGD", "one function"], tol=1e-3, seed=args.seed, **common)
    curves["Adaptive UQ-AGD ($M_k = 2Q_{m_k}$)"] = _batched_curve(res, est, tail)
    print(f"  ({time.perf_counter() - t0:.0f}s)")

    labels = list(curves)
    colors = dict(zip(labels, [SERIES[3], SERIES[4], SERIES[4], SERIES[4], SERIES[1]]))
    styles = {**dict(zip(labels[1:4], LSP_STYLES)), labels[0]: dict(ls="-.")}
    _reach_table(curves)

    def build(paper):
        fig = _draw(curves, colors, (6.5, 3.0) if paper else (9, 4.6), 1.2 if paper else 1.8,
                    title=None if paper else f"Acceleration without mini-batching (Markowitz, V = 0, {n_trials} trials)",
                    logx=True, styles=styles, xlabel=r"Gradient evaluations of $F$", ylabel=r"$\|x - x^*\|_\pi^2$")
        ax = fig.axes[0]
        top = 3.0
        ax.set_ylim(top=top)
        for label in labels[1:4]:
            if diverged[label]:
                # Mark where the run was stopped as divergent, at the top edge of the clipped axis.
                ax.plot(curves[label][0][-1], top, marker="x", ms=7, mew=1.5, color=colors[label], clip_on=False)
        ax.text(0.99, 0.97, r"$\times$ diverged", transform=ax.transAxes, ha="right", va="top",
                fontsize=7 if paper else 9, color="0.35")
        return fig

    _save(build, args.out / "ablation_batch1", args.paper, curves)


def ablation_increment(args):
    """Does growing by a whole total degree, rather than one function, explain the schedule gain? (V = 0)"""
    problem, basis, ref, tail = _markowitz()
    m_max, n_trials, n_sgd = _n_functions(10), args.cost_trials, 2 * 10**6
    common = dict(reference=ref, n_trials=n_trials)
    degree_of = {_n_functions(p): p for p in range(0, 41)}
    next_degree = lambda m: _n_functions(degree_of[m] + 1)  # noqa: E731
    curves = {}

    t0 = time.perf_counter()
    est = MonteCarloGradient(problem, basis, n_samples=_proportional_samples(basis))
    alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
    gd_step = gd_step_size(problem, basis, est)
    for unit, grow in [("one function", None), ("one degree", next_degree)]:
        check_every, tol = STALL.get(("ALSP", unit), (ALSP_CHECK_EVERY, ALSP_TOL))
        curves[f"ALSP, {unit}"] = _alsp(problem, basis, m_max, tail, common, args.seed + 5, n_sgd=n_sgd, grow=grow,
                                        check_every=check_every, tol=tol)
        for method, kw, K in [("UQ-GD", dict(step_size=gd_step), 3000), ("UQ-AGD", dict(alpha=alpha, beta=beta), 1500)]:
            res = adaptive_uq_descent(problem, basis, est, 3, m_max, K, check_every=STALL[method, unit], tol=1e-3,
                                      grow=grow, seed=args.seed, **kw, **common)
            curves[f"Adaptive {method}, {unit}"] = _batched_curve(res, est, tail)
    res = uq_accelerated_gradient_descent(est, _degree_schedule(final=10), 1200, alpha, beta, q=problem.q,
                                          seed=args.seed + 1, **common)
    curves["UQ-AGD, fixed schedule"] = _batched_curve(res, est, tail)
    print(f"  ({time.perf_counter() - t0:.0f}s)")
    _reach_table(curves)

    method_color = {"ALSP": SERIES[3], "Adaptive UQ-GD": SERIES[0], "Adaptive UQ-AGD": SERIES[1]}
    colors = {label: method_color[label.split(",")[0]] for label in curves if "," in label and "fixed" not in label}
    colors["UQ-AGD, fixed schedule"] = "0.45"
    styles = {label: dict(ls="--") for label in curves if label.endswith("one function")}
    styles["UQ-AGD, fixed schedule"] = dict(ls=":")
    _plot(curves, f"Growth increment under the stall rule (Markowitz, V = 0, {n_trials} trials)",
          args.out / "ablation_increment", colors, paper=args.paper, paper_size=(6.5, 3.0), logx=True,
          styles=styles, xlabel=r"Gradient evaluations of $F$", ylabel=r"$\|x - x^*\|_\pi^2$")


def alsp_sweep(args):
    """Sensitivity of ALSP to its stall-rule settings (V = 0), to make sure it is compared at its best.

    Selection rule, fixed before running: the setting reaching 1e-6 in the fewest gradient
    evaluations, ties broken at 1e-8. Writes alsp_sweep.md with every setting and the best
    setting at each accuracy level separately.
    """
    problem, basis, ref, tail = _markowitz()
    m_max, n_trials, n_sgd = _n_functions(10), args.sweep_trials, 4 * 10**6
    common = dict(reference=ref, n_trials=n_trials)
    levels = (1e-4, 1e-6, 1e-8)
    curves, rows = {}, []
    for check_every in (1000, 5000, 20000):
        for tol in (1e-2, 1e-3, 1e-4):
            label = f"check every {check_every}, tol {tol:g}"
            print(f" {label}")
            curve = _alsp(problem, basis, m_max, tail, common, args.seed + 5, n_sgd=n_sgd,
                          check_every=check_every, tol=tol)
            curves[label] = curve
            rows.append((label, check_every, tol, [_reach(curve[0], curve[1], t) for t in levels], curve[1][-1]))
    key = lambda r: tuple(np.nan_to_num(r[3][1:], nan=np.inf))  # noqa: E731  (1e-6, then 1e-8)
    best = min(rows, key=key)
    fmt = lambda v: f"{v:.1e}" if np.isfinite(v) else "-"  # noqa: E731
    lines = [f"# ALSP stall-rule sweep ({n_trials} trials, {n_sgd:.0e} steps)", "",
             "Gradient evaluations to first reach each error level.", "",
             "| Check every | Threshold | 1e-4 | 1e-6 | 1e-8 | Final error |", "|---|---|---|---|---|---|"]
    for label, ce, tol, reach, final in rows:
        mark = " **(selected)**" if label == best[0] else ""
        lines.append(f"| {ce} | {tol:g}{mark} | " + " | ".join(fmt(v) for v in reach) + f" | {final:.1e} |")
    per_level = [np.nanmin([r[3][i] for r in rows]) for i in range(len(levels))]
    lines += ["", "Best setting at each level separately: " + ", ".join(
        f"{t:.0e}: {fmt(v)}" for t, v in zip(levels, per_level)),
        "", f"Selected (fewest evaluations to 1e-6, ties at 1e-8): check every {best[1]}, threshold {best[2]:g}."]
    (args.out / "alsp_sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))
    best_setting = (best[1], best[2])

    colors = {label: SERIES[i % 3] for i, label in enumerate(curves)}
    styles = {label: LSP_STYLES[i // 3] for i, label in enumerate(curves)}
    _plot(curves, f"ALSP sensitivity to its stall rule (Markowitz, V = 0, {n_trials} trials)",
          args.out / "alsp_sweep", colors, paper=args.paper, paper_size=(6.5, 3.0), logx=True, styles=styles,
          xlabel=r"Gradient evaluations of $F$", ylabel=r"$\|x - x^*\|_\pi^2$")
    return best_setting


def revalidate(args):
    """Sweep ALSP's settings, adopt the selected one, then rerun the four comparison figures."""
    global ALSP_CHECK_EVERY, ALSP_TOL
    t_start = time.perf_counter()
    print("alsp_sweep:")
    ALSP_CHECK_EVERY, ALSP_TOL = alsp_sweep(args)
    print(f"  using ALSP check every {ALSP_CHECK_EVERY}, tol {ALSP_TOL:g} ({time.perf_counter() - t_start:.0f}s so far)")
    for name in ("comparison", "comparison_adaptive", "ablation_batch1", "ablation_increment"):
        print(f"{name}:")
        t0 = time.perf_counter()
        FIGURES[name](args)
        print(f"  ({time.perf_counter() - t0:.0f}s; {time.perf_counter() - t_start:.0f}s so far)")


def _best(rows):
    """Row with the fewest evaluations to 1e-6, ties broken at 1e-8."""
    return min(rows, key=lambda r: tuple(np.nan_to_num(r[1][1:], nan=np.inf)))


def sweep_extended(args):
    """Extend the stall-rule sweeps beyond the edges of the first grid, for every adaptive variant.

    Applies the same selection rule as ``alsp_sweep`` to each variant, writes sweep_extended.md,
    and returns the selected settings.
    """
    problem, basis, ref, tail = _markowitz()
    m_max, n_trials = _n_functions(10), args.sweep_trials
    common = dict(reference=ref, n_trials=n_trials)
    levels = (1e-4, 1e-6, 1e-8)
    degree_of = {_n_functions(p): p for p in range(0, 41)}
    next_degree = lambda m: _n_functions(degree_of[m] + 1)  # noqa: E731
    fmt = lambda v: f"{v:.1e}" if np.isfinite(v) else "-"  # noqa: E731
    grids = {
        ("ALSP", "one function"): [(250, 1e-2), (250, 1e-3), (500, 1e-2), (500, 1e-3), (1000, 1e-2)],
        ("ALSP", "one degree"): [(250, 1e-2), (500, 1e-2), (1000, 1e-2), (5000, 1e-2)],
    }
    est = MonteCarloGradient(problem, basis, n_samples=_proportional_samples(basis))
    alpha, beta = agd_parameters(problem.mu, 1 / problem.L)
    gd_step = gd_step_size(problem, basis, est)
    selected, lines = {}, [f"# Extended stall-rule sweep ({n_trials} trials)", "",
                           "Gradient evaluations to first reach each error level. The selected setting "
                           "(fewest evaluations to 1e-6, ties at 1e-8) is in bold.", ""]

    def report(variant, rows, setting_name):
        best = _best(rows)
        selected[variant] = best[0]
        lines.extend([f"## {variant[0]}, {variant[1]}", "", f"| {setting_name} | 1e-4 | 1e-6 | 1e-8 |", "|---|---|---|---|"])
        for setting, reach in rows:
            name = f"**{setting}**" if setting == best[0] else str(setting)
            lines.append(f"| {name} | " + " | ".join(fmt(v) for v in reach) + " |")
        lines.append("")
        print(f"  {variant}: selected {best[0]}")

    for variant, grid in grids.items():
        grow = next_degree if variant[1] == "one degree" else None
        n_sgd = 4 * 10**6 if grow is None else 2 * 10**6
        rows = []
        for check_every, tol in grid:
            print(f" {variant} check every {check_every}, tol {tol:g}")
            x, y, *_ = _alsp(problem, basis, m_max, tail, common, args.seed + 5, n_sgd=n_sgd, grow=grow,
                             check_every=check_every, tol=tol)
            rows.append(((check_every, tol), [_reach(x, y, t) for t in levels]))
        report(variant, rows, "(check every, threshold)")

    for unit, grow in [("one function", None), ("one degree", next_degree)]:
        for method, kw, K in [("UQ-GD", dict(step_size=gd_step), 3000), ("UQ-AGD", dict(alpha=alpha, beta=beta), 1500)]:
            rows = []
            for window in (2, 3, 5):
                t0 = time.perf_counter()
                res = adaptive_uq_descent(problem, basis, est, 3, m_max, K, check_every=window, tol=1e-3, grow=grow,
                                          seed=args.seed, **kw, **common)
                x, y, *_ = _batched_curve(res, est, tail)
                rows.append((window, [_reach(x, y, t) for t in levels]))
                print(f" {method}, {unit}, window {window}: {time.perf_counter() - t0:.0f}s")
            report((method, unit), rows, "Iterations between checks")

    (args.out / "sweep_extended.md").write_text("\n".join(lines) + "\n")
    print(f"  saved {args.out / 'sweep_extended.md'}")
    return selected


def revalidate_extended(args):
    """Run ``sweep_extended``, adopt the selected settings, and rerun only the figures they affect."""
    global ALSP_CHECK_EVERY, ALSP_TOL
    t_start = time.perf_counter()
    print("sweep_extended:")
    before = {("ALSP", "one function"): (ALSP_CHECK_EVERY, ALSP_TOL), **STALL}
    selected = sweep_extended(args)
    changed = {variant for variant, setting in selected.items() if before.get(variant) != setting}
    ALSP_CHECK_EVERY, ALSP_TOL = selected["ALSP", "one function"]
    STALL.update({variant: setting for variant, setting in selected.items() if variant != ("ALSP", "one function")})
    print(f"  changed: {sorted(changed) or 'none'} ({time.perf_counter() - t_start:.0f}s so far)")
    uses = {
        "comparison": {("ALSP", "one function")},
        "comparison_adaptive": {("ALSP", "one function"), ("UQ-GD", "one function"), ("UQ-AGD", "one function")},
        "ablation_batch1": {("ALSP", "one function"), ("UQ-AGD", "one function")},
        "ablation_increment": set(selected),
    }
    for name, deps in uses.items():
        if deps & changed:
            print(f"{name}:")
            t0 = time.perf_counter()
            FIGURES[name](args)
            print(f"  ({time.perf_counter() - t0:.0f}s; {time.perf_counter() - t_start:.0f}s so far)")
        else:
            print(f"{name}: unchanged settings, not rerun")


SECTION6 = {
    "fig1a": "6.1 - exact gradients (paper Fig. 1a)",
    "fig1b": "6.1 - Monte Carlo gradients, incl. GD with noise in v (Fig. 1b; absorbs old Fig. 2)",
    "fig3": "6.1 - fixed vs. growing basis (Fig. 3, schedule power 0.7)",
    "lemma31": "6.2 - Lemma 3.1 check (pair with markowitz)",
    "markowitz": "6.2 - GD/AGD convergence with noisy returns (pair with lemma31)",
    "markowitz_uq": "6.2 - distributions of optimal weights (full width) + markowitz_uq_table.tex",
    "comparison": "6.3 - comparison with Dong et al. and naive MC (full width)",
    "comparison_adaptive": "6.3 (optional) - same, with ALSP's stall rule for both UQ methods (full width)",
    "ablation_batch1": "6.3 (ablation) - accelerated SGD at batch size 1 diverges",
    "ablation_increment": "6.3 / appendix - stall rule with one-function vs. one-degree growth",
    "schedules": "6.4 - sensitivity to the growth schedule (full width)",
}


def section6(args):
    """Every figure of the proposed Section 6 layout, written to <out>/section6."""
    args.out = args.out / "section6"
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["# Section 6 figures", "", "| File | Placement |", "|---|---|"]
    for name, placement in SECTION6.items():
        print(f"{name}:")
        t0 = time.perf_counter()
        FIGURES[name](args)
        print(f"  ({time.perf_counter() - t0:.0f}s)")
        lines.append(f"| `{name}` | {placement} |")
    (args.out / "README.md").write_text("\n".join(lines) + "\n")


FIGURES = {
    "fig1a": fig1a, "fig1b": fig1b, "fig2": fig2, "fig3": fig3,
    "markowitz": markowitz, "lemma31": lemma31, "markowitz_uq": markowitz_uq, "cost": cost,
    "comparison": comparison, "comparison_adaptive": comparison_adaptive, "schedules": schedules,
    "ablation_batch1": ablation_batch1, "ablation_increment": ablation_increment, "alsp_sweep": alsp_sweep,
    "revalidate": revalidate, "sweep_extended": sweep_extended, "revalidate_extended": revalidate_extended,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figures", nargs="*", default=["all"], choices=[*FIGURES, "all", "section6"])
    parser.add_argument("--trials", type=int, default=200, help="Monte Carlo repetitions averaged per curve")
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--L", type=float, default=200.0)
    parser.add_argument("--power", type=float, default=0.7, help="fig3 schedule m_k = floor((k+10)^power) + 2")
    parser.add_argument("--fixed-m", type=int, default=None, help="fixed level for fig3 (default: max m_k reached)")
    parser.add_argument("--cost-trials", type=int, default=5,
                        help="trials for the gradient-evaluation comparisons (comparison and ablations)")
    parser.add_argument("--sweep-trials", type=int, default=5, help="trials per setting in alsp_sweep")
    parser.add_argument("--paper", action="store_true", help="also write half-width Computer Modern PDFs for LaTeX")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if "section6" in args.figures:
        section6(args)
        return
    names = list(FIGURES) if "all" in args.figures else args.figures
    for name in names:
        print(f"{name}:")
        FIGURES[name](args)


if __name__ == "__main__":
    main()
