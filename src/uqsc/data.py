"""Download and cache external data.

The weighted logistic regression benchmark of Dong, Grigas & Gupta (AISTATS 2025) uses
1000 rows of the LSAC National Longitudinal Bar Passage Study (Wightman, 1998), as
processed and published in their MIT-licensed repository. We download their processed
files at a pinned commit instead of redistributing them.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

DONG_REPO = "https://raw.githubusercontent.com/Cumberkid/Learning-the-Optimal-Solution-Path"
DONG_COMMIT = "cf499df19580e127cd0529620dec3af0d25f0ea3"
DONG_LOGISTIC = "data/reweighted-logistic-regression"


def cache_dir() -> Path:
    path = Path(os.environ.get("UQSC_CACHE", Path.home() / ".cache" / "uqsc"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch(relative_path: str) -> Path:
    """Path to a file from Dong et al.'s repository at the pinned commit, downloading it once."""
    target = cache_dir() / DONG_COMMIT[:7] / relative_path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{DONG_REPO}/{DONG_COMMIT}/{relative_path}", target)
    return target


def bar_passage() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Standardised features X (1000 x 45), labels y in {0, 1}, and feature names."""
    X_path, y_path = fetch(f"{DONG_LOGISTIC}/X_processed.csv"), fetch(f"{DONG_LOGISTIC}/y_processed.csv")
    names = X_path.read_text().splitlines()[0].split(",")
    X = np.loadtxt(X_path, delimiter=",", skiprows=1)
    y = np.loadtxt(y_path, delimiter=",", skiprows=1)
    return X, y, names


def bar_passage_reference_path() -> tuple[np.ndarray, np.ndarray]:
    """Dong et al.'s reference solution path, computed by 5000 iterations of gradient descent per
    grid point: lambda (1024 points on [0, 1], ascending) and the coefficients, shape (46, 1024),
    ordered as (weights, intercept) to match ``WeightedLogisticProblem``.

    Their file stores the intercept first and runs from lambda = 1 down to 0.
    """
    path = fetch(f"{DONG_LOGISTIC}/legendre/exact_soln_list_legendre.csv")
    table = np.loadtxt(path, delimiter=",", skiprows=1)[::-1, 1:]
    coefficients = np.hstack([table[:, 1:], table[:, :1]]).T
    return np.linspace(0.0, 1.0, len(table)), coefficients
