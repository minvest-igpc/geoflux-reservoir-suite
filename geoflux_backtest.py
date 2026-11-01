"""GeoFlux decline-model comparison, hindcast backtesting and flow diagnostics."""
import numpy as np
import pandas as pd

import geoflux_models as gm


def compare(fits):
    """Fit-quality table for {model: fit}. AICc penalises extra parameters;
    with short histories several models are usually statistically indistinguishable."""
    rows = []
    for name, f in fits.items():
        pn = gm.MODELS[name]["names"]
        score = f["aicc"] if np.isfinite(f["aicc"]) else f["aic"]
        rows.append({"Model": name, "Label": gm.label(name), "R2": f["r2"], "Resid_sd": f["resid_sd"],
                     "AICc": score, "At_bound": ", ".join(f["at_bound"]),
                     "Params": ", ".join(f"{n}={v:.4g}" for n, v in zip(pn, f["params"]))})
    df = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)
    df["dAICc"] = df["AICc"] - df["AICc"].min()
    return df


def default_origins(n):
    """Training sizes for rolling-origin hindcasts (each leaves 3+ points to test)."""
    start, stop = max(8, n // 2), n - 3
    if n < 12 or stop < start:
        return []
    return list(range(start, stop + 1, max(1, (stop - start) // 3)))


def _cum_boe(t_years, q):
    t, q = np.asarray(t_years, dtype=float), np.asarray(q, dtype=float)
    return float((0.5 * (q[1:] + q[:-1]) * np.diff(t)).sum() * gm.DAYS_PER_YEAR)


def backtest(t_years, q_boed, models, origins, terminal_pct=0.0):
    """Fit each model on the first k points and predict the rest. Errors are on
    log rate (0.1 is roughly 10%), plus the error in cumulative volume over the
    held-out period."""
    t, q = np.asarray(t_years, dtype=float), np.asarray(q_boed, dtype=float)
    rows = []
    for k in origins:
        for name in models:
            f = gm.fit(name, t[:k], q[:k])
            if f is None:
                continue
            pred = gm.rate(name, t[k:], f["params"], terminal_pct)
            if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
                continue
            e = np.log(pred) - np.log(q[k:])
            actual = _cum_boe(t[k - 1:], q[k - 1:])
            guess = _cum_boe(np.r_[t[k - 1], t[k:]], np.r_[q[k - 1], pred])
            rows.append({"Model": name, "Origin": int(k), "MAE_log": float(np.abs(e).mean()),
                         "Bias_log": float(e.mean()),
                         "Cum_error_pct": 100.0 * (guess / actual - 1.0) if actual > 0 else np.nan})
    return pd.DataFrame(rows, columns=["Model", "Origin", "MAE_log", "Bias_log", "Cum_error_pct"])


def backtest_summary(bt):
    if bt.empty:
        return pd.DataFrame(columns=["Model", "Label", "MAE_log", "Cum_error_pct", "Abs_cum_error_pct", "Origins"])
    g = bt.groupby("Model").agg(MAE_log=("MAE_log", "mean"), Cum_error_pct=("Cum_error_pct", "mean"),
                                Origins=("Origin", "count")).reset_index()
    g["Abs_cum_error_pct"] = bt.assign(a=bt["Cum_error_pct"].abs()).groupby("Model")["a"].mean().reindex(g["Model"]).to_numpy()
    g["Label"] = [gm.label(m) for m in g["Model"]]
    return g.sort_values("MAE_log").reset_index(drop=True)[
        ["Model", "Label", "MAE_log", "Cum_error_pct", "Abs_cum_error_pct", "Origins"]]


def loglog_slope(t_years, q_boed, window=8):
    """Slope of log rate against log time over the last `window` points. A slope
    near -0.5 is a hint of linear (transient) flow. It is only a hint: confirming
    a flow regime needs pressure data."""
    t, q = np.asarray(t_years, dtype=float)[-window:], np.asarray(q_boed, dtype=float)[-window:]
    ok = (t > 0) & (q > 0)
    if ok.sum() < 4:
        return {"slope": float("nan"), "hint": "Too few points for a diagnostic."}
    s = float(np.polyfit(np.log(t[ok]), np.log(q[ok]), 1)[0])
    if -0.65 <= s <= -0.35:
        hint = "Slope is near -1/2, which is consistent with linear (transient) flow."
    elif s > -0.35:
        hint = "Slope is flatter than linear flow: possible ramp-up or a rate-constrained period."
    else:
        hint = "Slope is steeper than linear flow: possible boundary influence, or noisy data."
    return {"slope": s, "hint": hint}


# END OF FILE: geoflux_backtest.py