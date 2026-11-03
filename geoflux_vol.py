"""GeoFlux volumetrics: probabilistic oil and gas in place.

OOIP (sm3) = A * h * phi * (1 - Sw) / Bo      OGIP (sm3) = A * h * phi * (1 - Sw) / Bg
with A in km2, h in m. Inputs are triangular (low, mode, high); porosity and
water saturation can be correlated (usually negatively). Recoverable volume is
in-place times a recovery factor range. These are resource estimates for screening,
not reserves.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import geoflux_risk as risk

SM3_TO_STB = 6.28981
SM3_TO_SCF = 35.3147

OIL_SPEC = {"area_km2": (8.0, 12.0, 18.0), "h_m": (15.0, 25.0, 40.0), "phi": (0.14, 0.18, 0.22),
            "sw": (0.20, 0.30, 0.45), "fvf": (1.25, 1.35, 1.45), "rf": (0.12, 0.25, 0.38)}
GAS_SPEC = {"area_km2": (6.0, 10.0, 16.0), "h_m": (20.0, 35.0, 60.0), "phi": (0.08, 0.12, 0.16),
            "sw": (0.20, 0.30, 0.45), "fvf": (0.0035, 0.0042, 0.0050), "rf": (0.60, 0.72, 0.85)}


def sample_volumetrics(spec, n, seed=0, phi_sw_corr=-0.3):
    """Triangular draws for every input. phi and Sw are linked by a Gaussian
    copula with the given correlation (negative: higher porosity, lower Sw)."""
    risk.check_spec(spec)
    rng = np.random.default_rng(seed)
    keys = list(spec)
    U = rng.random((n, len(keys)))
    z1 = rng.standard_normal(n)
    z2 = phi_sw_corr * z1 + np.sqrt(max(1.0 - phi_sw_corr ** 2, 0.0)) * rng.standard_normal(n)
    U[:, keys.index("phi")], U[:, keys.index("sw")] = norm.cdf(z1), norm.cdf(z2)
    return {k: risk._tri_ppf(U[:, i], *spec[k]) for i, k in enumerate(keys)}


def in_place(d):
    """Hydrocarbon in place (sm3) from draws or scalars: A h phi (1-Sw) / FVF."""
    return np.asarray(d["area_km2"]) * 1e6 * d["h_m"] * d["phi"] * (1.0 - np.asarray(d["sw"])) / d["fvf"]


def run(spec, n=10000, seed=0, phi_sw_corr=-0.3):
    draws = sample_volumetrics(spec, n, seed, phi_sw_corr)
    ip = in_place(draws)
    return {"draws": draws, "in_place": ip, "recoverable": ip * draws["rf"]}


def deterministic(spec):
    """In place and recoverable at the most-likely (mode) value of every input."""
    mode = {k: v[1] for k, v in spec.items()}
    ip = float(in_place(mode))
    return {"in_place": ip, "recoverable": ip * mode["rf"]}


def sensitivity(draws, output):
    """Spearman rank correlation of each input with the output, sorted by size.
    A quick guide to which input drives the uncertainty."""
    rows = [{"Input": k, "Rank_correlation": float(spearmanr(v, output)[0])} for k, v in draws.items()]
    df = pd.DataFrame(rows)
    return df.reindex(df["Rank_correlation"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# END OF FILE: geoflux_vol.py