"""GeoFlux type curves: per-well decline fits, monthly percentile curves and
analog-based forecasts for new (undrilled) wells.

Method: every analog well is fitted individually; a new well is treated as a
random draw from the analog population, scaled by lateral length. This assumes
the new well is exchangeable with its analogs: same formation and similar
completion design. Each analog's own fit uncertainty is not propagated.
"""
import numpy as np
import pandas as pd

import geoflux_models as gm


def fit_wells(series, model, terminal_pct=0.0, years=30, q_limit=0.0, min_points=12, min_r2=0.8):
    """Fit each well; returns ({well: fit + eur}, DataFrame of rejected wells)."""
    fits, rejected = {}, []
    for w, (t, q) in series.items():
        if len(t) < min_points:
            rejected.append((w, f"only {len(t)} points (need {min_points})"))
            continue
        f = gm.fit(model, t, q)
        if f is None:
            rejected.append((w, "fit failed"))
        elif f["r2"] < min_r2:
            rejected.append((w, f"R2 {f['r2']:.2f} below {min_r2}"))
        else:
            curve = gm.forecast_monthly(model, f["params"], years, terminal_pct)
            fits[w] = {**f, "eur": gm.eur_boe(curve, q_limit), "curve": curve, "points": len(t)}
    return fits, pd.DataFrame(rejected, columns=["Well", "Reason"])


def well_table(fits, meta):
    rows = [{"Well": w, "Points": f["points"], "R2": f["r2"], "EUR_MBOE": f["eur"] / 1e3} for w, f in fits.items()]
    df = pd.DataFrame(rows, columns=["Well", "Points", "R2", "EUR_MBOE"]).merge(meta, on="Well", how="left")
    if "Lateral_m" in df:
        df["EUR_MBOE_per_km"] = df["EUR_MBOE"] / (df["Lateral_m"] / 1000.0)
    return df


def pcurves(series, wells, lateral=None, min_wells=5, max_months=120):
    """Percentile curves by month on production. With `lateral` ({well: m}),
    rates are normalised to BOE/d per 1000 m. Percentiles are per month and
    SPE-style (P90 = low). Later months use fewer, older wells (see N)."""
    cols = {}
    for w in wells:
        t, q = series[w]
        s = pd.Series(q / lateral[w] * 1000.0 if lateral else q, index=np.round((t - 1.0 / 24.0) * 12.0).astype(int))
        cols[w] = s[~s.index.duplicated()]
    M = pd.DataFrame(cols).reindex(range(max_months))
    out = pd.DataFrame({"Month": M.index, "N": M.notna().sum(axis=1), "P90": M.quantile(0.10, axis=1),
                        "P50": M.quantile(0.50, axis=1), "P10": M.quantile(0.90, axis=1), "Mean": M.mean(axis=1)})
    return out[out["N"] >= min_wells].reset_index(drop=True)


def fit_type_curve(pc, model, column="P50"):
    """Fit a decline model to a percentile (or mean) curve so it can be extrapolated."""
    return gm.fit(model, (pc["Month"].to_numpy() + 0.5) / 12.0, pc[column].to_numpy())


def _scales(wells, lateral, lateral_new, beta):
    if lateral_new is None or lateral is None:
        return np.ones(len(wells))
    return np.array([(lateral_new / lateral[w]) ** beta for w in wells])


def analog_samples(fits, lateral=None, n=1000, lateral_new=None, beta=1.0, seed=0, with_rates=True):
    """Bootstrap analog wells as outcomes for a new well. Volumes scale with
    (lateral_new / lateral_analog) ** beta (beta = 1: proportional to length)."""
    wells = list(fits)
    idx = np.random.default_rng(seed).integers(0, len(wells), n)
    sc = _scales(wells, lateral, lateral_new, beta)[idx]
    eur = np.array([fits[w]["eur"] for w in wells])[idx] * sc
    out = {"eur": eur, "wells": [wells[i] for i in idx]}
    if with_rates:
        out["rates"] = np.vstack([fits[w]["curve"] for w in wells])[idx] * sc[:, None]
    return out


def loo_coverage(fits, lateral=None, beta=1.0):
    """Leave-one-out calibration: predict each well's EUR range (P90 to P10) from
    the OTHER wells, scaled to its lateral, and report how often the actual EUR
    falls inside. A calibrated 80% range should contain about 80% of wells."""
    wells = list(fits)
    rows = []
    for w in wells:
        others = [o for o in wells if o != w]
        eur = np.array([fits[o]["eur"] for o in others]) * _scales(others, lateral, lateral[w] if lateral else None, beta)
        lo, hi = np.percentile(eur, 10), np.percentile(eur, 90)
        rows.append({"Well": w, "EUR_MBOE": fits[w]["eur"] / 1e3, "P90_MBOE": lo / 1e3, "P10_MBOE": hi / 1e3,
                     "Inside": bool(lo <= fits[w]["eur"] <= hi)})
    df = pd.DataFrame(rows)
    return (float(df["Inside"].mean()) if len(df) else float("nan")), df


# END OF FILE: geoflux_typecurve.py