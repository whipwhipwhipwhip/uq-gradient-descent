"""Reproduce the figures and tables of Section 6 of the paper, and nothing else.

    uqsc-paper                  # everything, into results/paper/
    uqsc-paper fig1 table3      # selected items (table1 and table2 are aliases of fig4 and fig6)
    uqsc-paper --list           # what each item is, and the experiment that produces it

Each item runs one experiment from ``experiments.py`` with the settings used in the paper.
The raw outputs (PNG previews, plotted curves as .npz, logs) go to ``<out>/raw/`` under the
experiment's own name, and the files for LaTeX are copied to ``<out>/`` under the figure's or
table's number in the paper (fig1a.pdf, table1.tex, ...).
"""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import experiments as ex


@dataclass
class Item:
    caption: str
    run: object  # callable(args)
    outputs: dict[str, str]  # paper file name -> raw file name
    settings: dict = field(default_factory=dict)  # overrides of the uqsc-experiments defaults


def _logistic(args):
    """Fig. 6 / Table 2. kappa ~ 16, 60, 236: full protocol (tune ALSP and the stall window with
    --sweep-trials, rerun with 20 trials). kappa ~ 940: reduced protocol, as described in Section 6.3.1:
    settings carried over from kappa ~ 236, 5 trials, an ALSP budget of 6 Q_{m_max} kappa steps, then
    ALSP alone re-tuned over check intervals 4000 (carried over), 16000 and 64000."""
    args.ridges, args.cost_trials = [0.25, 1 / 16, 1 / 64], 20
    ex.logistic_kappa(args)
    args.ridges, args.cost_trials, args.logistic_settings, args.alsp_budget = [1 / 256], 5, (4000, 5), 6.0
    ex.logistic_kappa(args)
    args.alsp_checks = [16000, 64000]
    ex.logistic_alsp_tune(args)


ITEMS = {
    "fig1": Item("Fig. 1: quadratic (6.2), exact (a) and Monte Carlo (b) gradients, 200 trials",
                 lambda a: (ex.fig1a(a), ex.fig1b(a)),
                 {"fig1a.pdf": "fig1a.pdf", "fig1b.pdf": "fig1b.pdf"}, dict(trials=200)),
    "fig2": Item("Fig. 2: fixed vs. growing basis, m_k = floor((k+10)^0.7) + 2, 200 trials",
                 ex.fig3, {"fig2.pdf": "fig3.pdf"}, dict(trials=200, power=0.7)),
    "fig3": Item("Fig. 3: Markowitz, UQ-GD and UQ-AGD with noisy returns, 20 trials",
                 ex.markowitz, {"fig3.pdf": "markowitz.pdf"}, dict(trials=200)),
    "fig4": Item("Fig. 4 and Table 1: learned vs. true distribution of the optimal weights",
                 ex.markowitz_uq, {"fig4.pdf": "markowitz_uq.pdf", "table1.tex": "markowitz_uq_table.tex",
                                   "table1.md": "markowitz_uq_table.md"}),
    "fig5": Item("Fig. 5: comparison with LSP, ALSP and naive MC (Markowitz, V = 0), 5 trials",
                 ex.comparison, {"fig5.pdf": "comparison.pdf"}, dict(cost_trials=5)),
    "fig6": Item("Fig. 6 and Table 2: weighted logistic regression, speedup over ALSP against kappa",
                 _logistic, {"fig6.pdf": "logistic_kappa.pdf", "table2.tex": "logistic_speedup.tex",
                             "table2.md": "logistic_speedup.md"}, dict(sweep_trials=5)),
    "table3": Item("Table 3: gradient evaluations for three basis growth rates (Markowitz, V = 0), 5 trials",
                   ex.schedules, {"table3.tex": "schedules_table.tex", "table3.md": "schedules_table.md"},
                   dict(cost_trials=5)),
}
ALIASES = {"table1": "fig4", "table2": "fig6"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("items", nargs="*", metavar="item", help=f"default: all. One of: {', '.join([*ITEMS, *ALIASES])}")
    parser.add_argument("--out", type=Path, default=Path("results/paper"))
    parser.add_argument("--list", action="store_true", help="describe each item and exit")
    opts = parser.parse_args(argv)
    unknown = set(opts.items) - {*ITEMS, *ALIASES}
    if unknown:
        parser.error(f"unknown item(s): {', '.join(sorted(unknown))}")

    if opts.list:
        for name, item in ITEMS.items():
            print(f"{name:7s} {item.caption}\n        -> {', '.join(item.outputs)}")
        return

    raw = opts.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(ALIASES.get(n, n) for n in opts.items or ITEMS))
    t_start = time.perf_counter()
    for name in names:
        item = ITEMS[name]
        args = ex.build_parser().parse_args(["--paper", "--out", str(raw)])
        for key, value in item.settings.items():
            setattr(args, key, value)
        print(f"{name}: {item.caption}")
        t0 = time.perf_counter()
        item.run(args)
        for target, source in item.outputs.items():
            shutil.copyfile(raw / source, opts.out / target)
            print(f"  -> {opts.out / target}")
        print(f"  ({time.perf_counter() - t0:.0f}s; {time.perf_counter() - t_start:.0f}s so far)")


if __name__ == "__main__":
    main()
