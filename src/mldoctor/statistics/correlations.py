"""Pairwise-complete Pearson and Spearman correlations.

For every pair (a, b) only rows where *both* values are finite are used, and the
pair-specific sample size ``n`` is reported. Non-finite values (NaN, ±Inf) count as missing.

* Pearson r measures linear dependence; Spearman rho is Pearson r of average ranks and
  measures monotone dependence. Ties receive average ranks.
* A correlation is **undefined** (``None``) — not 0 — when either variable is constant on
  the pair's complete rows or when ``n`` is below the configured floor.
* Pearson is computed with masked sufficient statistics on mean-centred data (exact
  pairwise-complete values, vectorised).
* Spearman is exact for pairs without missing values (ranks of complete columns). For pairs
  involving missing values, ranks are recomputed on the pair's complete rows, which is exact
  but O(n log n) per pair; beyond ``max_exact_pairs`` such pairs fall back to ranks computed
  per column on its own observed values (the pandas convention) and are marked
  ``spearman_exact=False``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import rankdata

_REL_TOL = 1e-12


@dataclass(frozen=True)
class PairCorrelation:
    a: str
    b: str
    n: int
    pearson: float | None
    spearman: float | None
    undefined_reason: str | None = None
    spearman_exact: bool = True

    @property
    def max_abs(self) -> float | None:
        values = [abs(v) for v in (self.pearson, self.spearman) if v is not None]
        return max(values) if values else None


def _masked_pearson(
    x: NDArray[np.float64], mask: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Return (r matrix with NaN where undefined, pair counts, constant-on-subset flags)."""
    m = mask.astype(float)
    observed = m.sum(axis=0)
    totals = np.where(mask, x, 0.0).sum(axis=0)
    col_means = np.divide(totals, observed, out=np.zeros_like(totals), where=observed > 0)
    xc = np.where(mask, x - col_means, 0.0)
    n = m.T @ m
    s = xc.T @ m  # s[a, b] = sum of centred a over rows where a and b observed
    ss = (xc**2).T @ m  # ss[a, b] = sum of squares of centred a over joint rows
    sab = xc.T @ xc
    with np.errstate(invalid="ignore", divide="ignore"):
        var_a = ss - s**2 / n  # n * variance of a on joint rows
        var_b = var_a.T
        cov = sab - s * s.T / n
        scale_a = ss / n
        constant = (var_a <= _REL_TOL * np.maximum(scale_a, 1e-300) * n) | (
            var_b <= _REL_TOL * np.maximum(scale_a.T, 1e-300) * n
        )
        r = cov / np.sqrt(var_a * var_b)
    r = np.clip(r, -1.0, 1.0)
    r[constant | (n == 0)] = np.nan
    return r, n, constant


def _rank_columns(x: NDArray[np.float64], mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    ranks = np.full_like(x, np.nan)
    for j in range(x.shape[1]):
        col = mask[:, j]
        if col.any():
            ranks[col, j] = rankdata(x[col, j])
    return ranks


def _pair_spearman(xa: NDArray[np.float64], xb: NDArray[np.float64]) -> float | None:
    ra = rankdata(xa)
    rb = rankdata(xb)
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return None
    return float(np.clip(np.corrcoef(ra, rb)[0, 1], -1.0, 1.0))


def pairwise_correlations(
    frame: pd.DataFrame, *, min_n: int = 10, max_exact_pairs: int = 20000
) -> list[PairCorrelation]:
    """All pairs i < j of the numeric columns of ``frame``, in column order."""
    columns = [str(c) for c in frame.columns]
    p = len(columns)
    if p < 2:
        return []
    x = frame.to_numpy(dtype=float, na_value=np.nan)
    mask = np.isfinite(x)
    x = np.where(mask, x, np.nan)

    pearson, counts, constant = _masked_pearson(np.nan_to_num(x), mask)

    ranks = _rank_columns(x, mask)
    spearman_approx, _, _ = _masked_pearson(np.nan_to_num(ranks), mask)
    complete = mask.all(axis=0)

    results: list[PairCorrelation] = []
    exact_budget = max_exact_pairs
    for i in range(p):
        for j in range(i + 1, p):
            n = int(counts[i, j])
            if n < min_n:
                results.append(
                    PairCorrelation(columns[i], columns[j], n, None, None, "insufficient_pairs")
                )
                continue
            if constant[i, j]:
                results.append(
                    PairCorrelation(columns[i], columns[j], n, None, None, "constant_on_pair")
                )
                continue
            r = float(pearson[i, j])
            exact = True
            if complete[i] and complete[j]:
                rho: float | None = float(spearman_approx[i, j])
            elif exact_budget > 0:
                joint = mask[:, i] & mask[:, j]
                rho = _pair_spearman(x[joint, i], x[joint, j])
                exact_budget -= 1
            else:
                value = spearman_approx[i, j]
                rho = None if np.isnan(value) else float(value)
                exact = False
            results.append(
                PairCorrelation(columns[i], columns[j], n, r, rho, None, spearman_exact=exact)
            )
    return results


def pearson_spearman(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[
    int, float | None, float | None
]:
    """Pairwise-complete (n, Pearson, Spearman) for two vectors; ``None`` when undefined."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    joint = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[joint], ya[joint]
    n = int(xa.size)
    if n < 2 or np.ptp(xa) == 0 or np.ptp(ya) == 0:
        return n, None, None
    r = float(np.clip(np.corrcoef(xa, ya)[0, 1], -1.0, 1.0))
    return n, r, _pair_spearman(xa, ya)
