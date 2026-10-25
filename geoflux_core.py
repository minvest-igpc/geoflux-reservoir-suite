"""GeoFlux shared seismology utilities (domain-neutral).

Used by the mine module now and by the oil & gas module later.
All statistics here are standard, documented methods; none are tuned to a site.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- magnitudes
def energy_to_magnitude(energy_j):
    """Gutenberg-Richter energy-magnitude relation: log10(E[J]) = 1.5*M + 4.8.
    A rough conversion. Prefer the magnitude your monitoring system reports."""
    return (np.log10(np.maximum(np.asarray(energy_j, dtype=float), 1e-12)) - 4.8) / 1.5


def magnitude_to_energy(m):
    return 10.0 ** (1.5 * np.asarray(m, dtype=float) + 4.8)


def detect_mag_bin(m):
    """Return 0.1 if magnitudes look rounded to 0.1 (common in catalogues), else 0.0."""
    m = np.asarray(m, dtype=float)
    m = m[np.isfinite(m)]
    if m.size < 30:
        return 0.0
    return 0.1 if np.mean(np.abs(m / 0.1 - np.round(m / 0.1)) < 1e-6) > 0.95 else 0.0


# ------------------------------------------------- completeness and b-value
def estimate_mc_maxc(m, mag_bin=0.0, correction=0.2):
    """Magnitude of completeness by maximum curvature (modal histogram bin)
    plus a conservative correction (Woessner & Wiemer, 2005).
    Returns None when there are too few events."""
    m = np.asarray(m, dtype=float)
    m = m[np.isfinite(m)]
    if m.size < 20:
        return None
    hb = mag_bin if mag_bin > 0 else 0.1
    if mag_bin > 0:   # bins centred on the rounded magnitude grid
        edges = np.arange(m.min() - hb / 2.0, m.max() + hb, hb)
    else:
        edges = np.arange(np.floor(m.min() / hb) * hb, m.max() + 2 * hb, hb)
    counts, edges = np.histogram(m, bins=edges)
    centre = edges[int(np.argmax(counts))] + hb / 2.0
    return float(np.round(centre + correction, 6))


def estimate_b_value(m, mc, mag_bin=0.0, min_events=50):
    """Aki (1965) maximum-likelihood b-value above Mc, with the Utsu
    correction for magnitudes rounded to mag_bin:

        b = log10(e) / (mean(M) - (Mc - mag_bin/2)),   sigma_b = b / sqrt(N)

    Returns None if fewer than min_events events lie at or above Mc."""
    m = np.asarray(m, dtype=float)
    m = m[np.isfinite(m)]
    sel = m[m >= mc - 1e-9]
    if sel.size < min_events:
        return None
    denom = sel.mean() - (mc - mag_bin / 2.0)
    if denom <= 0:
        return None
    b = float(np.log10(np.e) / denom)
    return {"b": b, "sigma": b / np.sqrt(sel.size), "n": int(sel.size), "mc": float(mc)}


def b_value_series(times, mags, mc, mag_bin=0.0, window=150, step=25):
    """Moving b-value over windows of a fixed number of events above Mc.
    Event-count windows keep the statistical precision constant when the
    event rate changes."""
    times = np.asarray(times)
    mags = np.asarray(mags, dtype=float)
    order = np.argsort(times)
    t, m = times[order], mags[order]
    keep = m >= mc - 1e-9
    t, m = t[keep], m[keep]
    rows = []
    for i in range(0, len(m) - window + 1, step):
        r = estimate_b_value(m[i:i + window], mc, mag_bin, min_events=window)
        if r:
            rows.append({"Time": t[i + window // 2], "b": r["b"],
                         "sigma": r["sigma"], "N": window})
    return pd.DataFrame(rows, columns=["Time", "b", "sigma", "N"])


# ------------------------------------------------------- exceedance (Poisson)
def exceedance_probability(rate_per_day, b, mc, m_star, horizon_days):
    """P(at least one event >= m_star within horizon_days), assuming a
    stationary Poisson process with a Gutenberg-Richter magnitude law:

        N = rate * T * 10^(-b (M* - Mc)),   P = 1 - exp(-N)

    Clustered seismicity violates the Poisson assumption, so treat the
    result as a screening number, not a forecast."""
    m_star = max(float(m_star), float(mc))
    n_expected = rate_per_day * horizon_days * 10.0 ** (-b * (m_star - mc))
    return float(1.0 - np.exp(-n_expected))