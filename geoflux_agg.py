"""GeoFlux portfolio aggregation: two clearly separate views.

1. Arithmetic sum by category (P90 + P90, P50 + P50, P10 + P10): the reporting
   convention. The low case is conservative and the high case optimistic.
2. Probabilistic aggregation of the simulated samples with a correlation
   input: the investment-risk view that captures the portfolio effect.

Means add exactly in both views. Percentiles only add under perfect correlation.
"""
import numpy as np
import pandas as pd


def percentiles(x):
    """SPE convention: P90 is the value exceeded with 90% probability (the 10th
    percentile, a LOW case); P10 is the HIGH case (90th percentile)."""
    x = np.asarray(x, dtype=float)
    return {"P90": float(np.percentile(x, 10)), "P50": float(np.percentile(x, 50)),
            "P10": float(np.percentile(x, 90)), "Mean": float(x.mean())}


def well_table(samples):
    rows = [{"Well": name, **percentiles(x)} for name, x in samples.items()]
    return pd.DataFrame(rows, columns=["Well", "P90", "P50", "P10", "Mean"])


def arithmetic_sum(table):
    return {k: float(table[k].sum()) for k in ["P90", "P50", "P10", "Mean"]}


def exchangeable_matrix(k, rho):
    c = np.full((k, k), float(rho))
    np.fill_diagonal(c, 1.0)
    return c


def correlated_samples(X, rank_corr, seed=0):
    """Reorder each column of X so the columns have the target Spearman rank
    correlation. Each column keeps exactly its own values (marginals preserved);
    the original ordering is discarded, so inputs may arrive already correlated."""
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    C = np.asarray(rank_corr, dtype=float)
    if C.size == 1:
        C = exchangeable_matrix(k, float(C))
    if C.shape != (k, k) or not np.allclose(C, C.T) or not np.allclose(np.diag(C), 1.0):
        raise ValueError("Correlation must be a number or a symmetric matrix with ones on the diagonal.")
    Cz = 2.0 * np.sin(np.pi * C / 6.0)          # Spearman -> Gaussian-copula correlation
    np.fill_diagonal(Cz, 1.0)
    w, V = np.linalg.eigh(Cz)
    if w.min() < -1e-8:
        raise ValueError("The correlation matrix is not positive semi-definite.")
    L = V * np.sqrt(np.maximum(w, 0.0))
    Z = np.random.default_rng(seed).standard_normal((n, k)) @ L.T
    out = np.empty_like(X)
    for j in range(k):
        ranks = np.argsort(np.argsort(Z[:, j]))
        out[:, j] = np.sort(X[:, j])[ranks]
    return out


def aggregate(samples, rank_corr=0.0, seed=0):
    """samples: {name: 1-D array of equal length}. Returns both views."""
    names = list(samples)
    if len({len(v) for v in samples.values()}) != 1:
        raise ValueError("All wells need the same number of samples.")
    X = np.column_stack([np.asarray(samples[n], dtype=float) for n in names])
    table = well_table(samples)
    arith = arithmetic_sum(table)
    total = correlated_samples(X, rank_corr, seed).sum(axis=1)
    prob = percentiles(total)
    effect = {k: (100.0 * (prob[k] / arith[k] - 1.0) if arith[k] else float("nan")) for k in ["P90", "P50", "P10"]}
    return {"wells": table, "arithmetic": arith, "probabilistic": prob,
            "difference_pct": effect, "total_samples": total}


# END OF FILE: geoflux_agg.py