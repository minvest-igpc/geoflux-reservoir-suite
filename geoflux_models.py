"""GeoFlux decline-model library: modified hyperbolic Arps, Duong, power-law
exponential (PLE) and stretched exponential (SEPD), with fitting and
parameter-uncertainty sampling.

Time is in YEARS and rate in BOE/d everywhere. Duong's model is defined in days,
so it converts internally. Standard published forms only; nothing is site-calibrated.
"""
import numpy as np
from scipy.optimize import curve_fit

import geoflux_dca as dca

DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = 30.4375

MODELS = {
    "arps": {"label": "Arps hyperbolic (modified)", "names": ("qi", "di", "b"),
             "lo": (1e-6, 1e-4, 0.05), "hi": (np.inf, 50.0, 2.0)},
    "duong": {"label": "Duong", "names": ("q1", "a", "m"),
              "lo": (1e-6, 1e-3, 1.001), "hi": (np.inf, 5.0, 2.5)},
    "ple": {"label": "Power-law exponential", "names": ("qi", "dinf", "dhat", "n"),
            "lo": (1e-6, 0.0, 1e-4, 0.05), "hi": (np.inf, 5.0, 50.0, 0.95)},
    "sepd": {"label": "Stretched exponential", "names": ("q0", "tau", "n"),
             "lo": (1e-6, 1e-4, 0.05), "hi": (np.inf, 100.0, 1.0)},
}


def label(model):
    return MODELS[model]["label"]


def rate(model, t_years, p, terminal_pct=0.0):
    """Rate (BOE/d) at times t_years for parameters p. terminal_pct (effective
    %/year) only applies to the Arps model; the others are already bounded."""
    t = np.asarray(t_years, dtype=float)
    if model == "arps":
        dmin = -np.log(1.0 - terminal_pct / 100.0) if terminal_pct > 0 else None
        return dca.arps_rate(t, p[0], p[1], p[2], dmin)
    if model == "duong":
        q1, a, m = p
        td = np.maximum(t * DAYS_PER_YEAR, 1e-9)
        return q1 * td ** (-m) * np.exp(a / (1.0 - m) * (td ** (1.0 - m) - 1.0))
    if model == "ple":
        qi, dinf, dhat, n = p
        return qi * np.exp(-dinf * t - dhat * np.maximum(t, 0.0) ** n)
    if model == "sepd":
        q0, tau, n = p
        return q0 * np.exp(-(np.maximum(t, 0.0) / tau) ** n)
    raise KeyError(model)


def _starts(model, t, q):
    q0, t0 = float(q[0]), max(float(t[0]), 1e-3)
    if model == "arps":
        return [[q0 * 1.1, 1.0, 1.0], [q0 * 1.3, 2.5, 1.2], [q0 * 1.05, 0.5, 0.6]]
    if model == "duong":
        out = []
        for a, m in [(1.0, 1.1), (0.8, 1.2), (1.5, 1.05)]:
            shape = float(rate("duong", t0, [1.0, a, m]))
            if np.isfinite(shape) and shape > 0:
                out.append([q0 / shape, a, m])
        return out
    if model == "ple":
        return [[q0 * 1.2, 0.05, 1.0, 0.5], [q0 * 1.5, 0.02, 2.0, 0.3], [q0 * 1.1, 0.1, 0.5, 0.7]]
    return [[q0 * 1.2, 0.5, 0.5], [q0 * 1.5, 0.2, 0.4], [q0 * 1.1, 1.0, 0.7]]


def fit(model, t_years, q_boed, b_bounds=None):
    """Least-squares fit in log-rate space with several starting points.
    Returns a dict (params, covariance, AIC...) or None if every start fails."""
    spec = MODELS[model]
    lo, hi = list(spec["lo"]), list(spec["hi"])
    if model == "arps" and b_bounds is not None:
        lo[2], hi[2] = b_bounds
    t = np.asarray(t_years, dtype=float)
    q = np.asarray(q_boed, dtype=float)
    ok = np.isfinite(t) & np.isfinite(q) & (q > 0) & (t > 0)
    t, q = t[ok], q[ok]
    k = len(lo)
    if len(t) < max(6, k + 3):
        return None
    lq = np.log(q)

    def f(tt, *p):
        return np.log(np.maximum(rate(model, tt, p), 1e-300))

    best = None
    for p0 in _starts(model, t, q):
        p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        try:
            popt, pcov = curve_fit(f, t, lq, p0=p0, bounds=(lo, hi), maxfev=40000)
        except Exception:
            continue
        rss = float(((lq - f(t, *popt)) ** 2).sum())
        if np.isfinite(rss) and (best is None or rss < best[0]):
            best = (rss, popt, pcov)
    if best is None:
        return None
    rss, popt, pcov = best
    n = len(t)
    rss_s = max(rss, 1e-12)
    aic = n * np.log(rss_s / n) + 2 * k
    ss_tot = float(((lq - lq.mean()) ** 2).sum())
    at_bound = [nm for nm, v, l, h in zip(spec["names"], popt, lo, hi)
                if (np.isfinite(l) and abs(v - l) <= 1e-3 * max(1.0, abs(l)) + 1e-9)
                or (np.isfinite(h) and abs(v - h) <= 1e-3 * max(1.0, abs(h)))]
    return {"model": model, "params": [float(v) for v in popt], "pcov": pcov, "lo": lo, "hi": hi,
            "rss": rss, "n": n, "k": k, "aic": float(aic),
            "aicc": float(aic + 2 * k * (k + 1) / (n - k - 1)) if n > k + 1 else float("nan"),
            "r2": 1.0 - rss / ss_tot if ss_tot > 0 else float("nan"),
            "resid_sd": float(np.sqrt(rss / max(n - k, 1))), "at_bound": at_bound}


def sample_params(fit_, n, rng):
    """Draw parameter sets from the fit covariance, keeping only draws inside
    the valid bounds (falls back to a +/-15% spread if the covariance is unusable)."""
    p = np.array(fit_["params"])
    lo, hi = np.array(fit_["lo"], dtype=float), np.array(fit_["hi"], dtype=float)
    cov = fit_.get("pcov")
    if cov is not None and np.all(np.isfinite(cov)):
        d = rng.multivariate_normal(p, cov, size=5 * n, check_valid="ignore")
    else:
        d = p * rng.lognormal(0.0, 0.15, size=(5 * n, len(p)))
    inside = np.all((d >= lo) & (d <= hi), axis=1)
    good = d[inside]
    if len(good) >= n:
        return good[:n]
    fill = np.clip(d[~inside][: n - len(good)], lo, hi)
    return np.vstack([good, fill])[:n]


def forecast_monthly(model, p, years, terminal_pct=0.0):
    """Monthly mid-point rates (BOE/d) from first production."""
    t = (np.arange(int(round(years * 12))) + 0.5) / 12.0
    return rate(model, t, p, terminal_pct)


def eur_boe(q_monthly, q_limit=0.0):
    """Cumulative volume (BOE) until the rate first falls below q_limit."""
    q = np.asarray(q_monthly, dtype=float)
    below = np.flatnonzero(q < q_limit)
    n = int(below[0]) if below.size else len(q)
    return float(q[:n].sum() * DAYS_PER_MONTH)


def probabilistic(model, fit_, n, years, terminal_pct=0.0, q_limit=0.0, seed=0):
    """Monte Carlo over parameter uncertainty. Keeps every sample (parameters,
    monthly rates and EUR) so results can be aggregated correctly later."""
    rng = np.random.default_rng(seed)
    draws = sample_params(fit_, n, rng)
    R = np.array([forecast_monthly(model, p, years, terminal_pct) for p in draws])
    good = np.all(np.isfinite(R), axis=1)
    draws, R = draws[good], R[good]
    eur = np.array([eur_boe(r, q_limit) for r in R])
    return {"params": draws, "rates": R, "eur": eur}


# END OF FILE: geoflux_models.py