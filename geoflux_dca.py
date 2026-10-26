"""GeoFlux oil & gas: decline-curve analysis and simple discounted cash flow.

Arps decline with a terminal-decline switch, a least-squares fit with parameter
uncertainty, and an economics model driven only by the inputs the user supplies.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

DAYS_PER_MONTH = 30.4375


def arps_rate(t_years, qi, di, b, dmin=None):
    """Arps hyperbolic decline q = qi / (1 + b*di*t)^(1/b) with t in years and
    di nominal per year. With dmin (nominal per year) the curve switches to
    exponential decline once the instantaneous decline falls to dmin
    (modified hyperbolic), which prevents unphysical long tails."""
    t = np.asarray(t_years, dtype=float)
    if b < 1e-6:
        q = qi * np.exp(-di * t)
    else:
        q = qi / (1.0 + b * di * t) ** (1.0 / b)
    if dmin is not None and 0 < dmin < di and b >= 1e-6:
        t_sw = (di / dmin - 1.0) / (b * di)
        q_sw = qi / (1.0 + b * di * t_sw) ** (1.0 / b)
        q = np.where(t > t_sw, q_sw * np.exp(-dmin * (t - t_sw)), q)
    return q


def production_time_axis(dates):
    """Years since first record, with a half-month offset (monthly average rates)."""
    d = pd.Series(dates)
    return ((d - d.min()).dt.days / 365.25 + 1.0 / 24.0).to_numpy()


def fit_arps(t_years, q_boed, b_bounds=(0.05, 2.0)):
    """Least-squares Arps fit in log-rate space. Returns None if it fails."""
    t = np.asarray(t_years, dtype=float)
    q = np.asarray(q_boed, dtype=float)
    ok = np.isfinite(t) & np.isfinite(q) & (q > 0)
    t, q = t[ok], q[ok]
    if len(t) < 6:
        return None
    lq = np.log(q)

    def model(tt, qi, di, b):
        return np.log(qi) - np.log1p(b * di * tt) / b

    try:
        popt, pcov = curve_fit(model, t, lq, p0=[q[0] * 1.1, 1.0, 1.0],
                               bounds=([1e-6, 1e-4, b_bounds[0]], [np.inf, 50.0, b_bounds[1]]),
                               maxfev=20000)
    except Exception:
        return None
    res = lq - model(t, *popt)
    ss_res, ss_tot = float((res ** 2).sum()), float(((lq - lq.mean()) ** 2).sum())
    return {"qi": float(popt[0]), "di": float(popt[1]), "b": float(popt[2]), "pcov": pcov,
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "resid_sd": float(np.sqrt(ss_res / max(len(t) - 3, 1))), "n": int(len(t)),
            "b_at_bound": bool(min(abs(popt[2] - b_bounds[0]), abs(popt[2] - b_bounds[1])) < 1e-3)}


def sample_arps_params(fit, n, rng, b_bounds=(0.05, 2.0)):
    """Draw (qi, di, b) from the fit covariance; falls back to +/-15% lognormal
    spread if the covariance is unusable. Draws are clipped to valid ranges."""
    mean = np.array([fit["qi"], fit["di"], fit["b"]])
    cov = fit.get("pcov")
    if cov is not None and np.all(np.isfinite(cov)):
        draws = rng.multivariate_normal(mean, cov, size=n, check_valid="ignore")
    else:
        draws = mean * rng.lognormal(0.0, 0.15, size=(n, 3))
    draws[:, 0] = np.clip(draws[:, 0], mean[0] * 0.05, mean[0] * 20)
    draws[:, 1] = np.clip(draws[:, 1], 1e-3, 50.0)
    draws[:, 2] = np.clip(draws[:, 2], b_bounds[0], b_bounds[1])
    return draws


def forecast_monthly(qi, di, b, dmin_eff_pct, years):
    """Monthly mid-point rates (BOE/d) from first production."""
    months = int(round(years * 12))
    t = (np.arange(months) + 0.5) / 12.0
    dmin = -np.log(1.0 - dmin_eff_pct / 100.0) if dmin_eff_pct > 0 else None
    return arps_rate(t, qi, di, b, dmin)


def _irr_monthly(cf, capex):
    if len(cf) == 0 or cf.sum() <= capex:
        return None
    t = np.arange(1, len(cf) + 1)

    def f(r):
        return -capex + float((cf / (1.0 + r) ** t).sum())
    try:
        if f(1e-9) <= 0 or f(5.0) >= 0:
            return None
        return float(brentq(f, 1e-9, 5.0))
    except Exception:
        return None


def economics(q_bpd, price, royalty_pct, var_opex, fixed_opex_month, capex,
              disc_rate_pct, tax_pct=0.0, min_rate=0.0):
    """Simple monthly discounted cash flow (any consistent currency). Production
    stops at the first month with negative operating cash flow or a rate below
    min_rate. Tax applies to positive operating cash flow only (no depreciation)."""
    q = np.asarray(q_bpd, dtype=float)
    vol = q * DAYS_PER_MONTH
    opcf = vol * (price * (1.0 - royalty_pct / 100.0) - var_opex) - fixed_opex_month
    bad = np.flatnonzero((opcf < 0) | (q < min_rate))
    n = int(bad[0]) if bad.size else len(q)
    vol, opcf = vol[:n], opcf[:n]
    cf = opcf - np.maximum(opcf, 0.0) * tax_pct / 100.0
    t = np.arange(1, n + 1)
    r_m = (1.0 + disc_rate_pct / 100.0) ** (1.0 / 12.0) - 1.0
    npv = float(-capex + (cf * (1.0 + r_m) ** (-t)).sum())
    pb = np.flatnonzero(np.cumsum(cf) >= capex)
    irr_m = _irr_monthly(cf, capex)
    return {"eur_boe": float(vol.sum()), "npv": npv, "life_months": n, "cf": cf,
            "payback_months": int(pb[0]) + 1 if pb.size else None,
            "irr_annual": (1.0 + irr_m) ** 12 - 1.0 if irr_m is not None else None}


def breakeven_price(q_bpd, royalty_pct, var_opex, fixed_opex_month, capex,
                    disc_rate_pct, tax_pct=0.0, min_rate=0.0, hi=1000.0):
    """Price per BOE at which NPV = 0. None if even `hi` is not enough."""
    def f(p):
        return economics(q_bpd, p, royalty_pct, var_opex, fixed_opex_month, capex,
                         disc_rate_pct, tax_pct, min_rate)["npv"]
    if f(hi) < 0:
        return None
    if f(1e-6) >= 0:
        return 0.0
    return float(brentq(f, 1e-6, hi))


def spe_percentiles(values):
    """SPE convention: P90 is the value exceeded with 90% probability (a LOW
    case, the 10th percentile); P10 is the HIGH case (90th percentile)."""
    v = np.asarray(values, dtype=float)
    return {"P90 (low)": float(np.percentile(v, 10)), "P50": float(np.percentile(v, 50)),
            "P10 (high)": float(np.percentile(v, 90))}


# END OF FILE: geoflux_dca.py