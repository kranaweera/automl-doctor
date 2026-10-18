"""Synthetic datasets with known statistical properties (deterministic)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_frame(n: int = 400, seed: int = 0, prefix: str = "r") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.uniform(0, 10, size=n)
    y = 3 * x1 - 2 * x2 + 0.5 * x3 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame(
        {"row_id": [f"{prefix}{i}" for i in range(n)], "x1": x1, "x2": x2, "x3": x3, "y": y}
    )


def classification_frame(
    n: int = 600, seed: int = 0, prefix: str = "c", minority_fraction: float = 0.4
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < minority_fraction).astype(int)
    x1 = rng.normal(loc=y * 1.0, size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    return pd.DataFrame(
        {"row_id": [f"{prefix}{i}" for i in range(n)], "x1": x1, "x2": x2, "cat": cat, "y": y}
    )


def grouped_frame(n_groups: int = 40, per_group: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"g{i:03d}" for i in range(n_groups)], per_group)
    effect = np.repeat(rng.normal(scale=2.0, size=n_groups), per_group)
    x = rng.normal(size=n_groups * per_group)
    y = 2 * x + effect + rng.normal(scale=0.3, size=n_groups * per_group)
    return pd.DataFrame(
        {"row_id": [f"r{i}" for i in range(len(x))], "pid": groups, "x": x, "y": y}
    )
