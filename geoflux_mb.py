"""GeoFlux material balance (SI units): P/Z for gas and Havlena-Odeh for oil.

Gas:  p/z = (pi/zi) (1 - Gp/G)  ->  straight line in p/z against Gp; G is the
      intercept on the Gp axis. Assumes volumetric depletion (no water influx).
Oil:  F = N (Eo + m Eg + Efw) + We, with F the underground withdrawal. Three
      straight-line forms are provided: depletion, unknown gas cap, and a
      steady-state (Schilthuis) aquifer.
Volumes in sm3 (standard) and rm3 (reservoir); pressure in MPa; time in days.
"""
import numpy as np

import geoflux_pvt as pv


def _mvn(mean, cov, n, rng):
    return rng.multivariate_normal(mean, cov, size=n, check_valid="ignore")


def pz_analysis(gp, p_mpa, z, p_ab=None, z_ab=None, n=5000, seed=0):
    """Straight-line P/Z analysis. Returns the gas initially in place G (sm3),
    its sampled uncertainty from the regression covariance, and optionally the
    recoverable volume down to an abandonment pressure."""
    gp, p, z = (np.asarray(a, dtype=float) for a in (gp, p_mpa, z))
    pz = p / z
    if len(gp) < 4 or np.ptp(gp) <= 0:
        raise ValueError("Need at least 4 points with increasing cumulative production.")
    (b, a), cov = np.polyfit(gp, pz, 1, cov=True)
    fitted = a + b * gp
    ss_tot = float(((pz - pz.mean()) ** 2).sum())
    out = {"a": float(a), "b": float(b), "r2": 1.0 - float(((pz - fitted) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan"),
           "n": len(gp), "G": float(-a / b) if b < 0 < a else float("nan")}
    h = len(gp) // 2
    s_early, s_late = np.polyfit(gp[: h + 1], pz[: h + 1], 1)[0], np.polyfit(gp[h:], pz[h:], 1)[0]
    out["slope_ratio"] = float(s_late / s_early) if s_early != 0 else float("nan")
    out["curved"] = bool(np.isfinite(out["slope_ratio"]) and abs(out["slope_ratio"] - 1.0) > 0.08)
    for key, sl in (("G_early", slice(0, h + 1)), ("G_late", slice(h, None))):      # straight-line brackets
        (bb, aa) = np.polyfit(gp[sl], pz[sl], 1)
        out[key] = float(-aa / bb) if bb < 0 < aa else float("nan")
    d = _mvn([b, a], cov, n, np.random.default_rng(seed))
    ok = (d[:, 0] < 0) & (d[:, 1] > 0)
    bs, as_ = d[ok, 0], d[ok, 1]
    out["G_samples"] = -as_ / bs
    if p_ab is not None and z_ab:
        gp_ab = np.clip((p_ab / z_ab - as_) / bs, 0.0, out["G_samples"])
        out["recoverable_samples"] = gp_ab
    return out


def ho_terms(p, pvt, pi, swi=0.2, cw=4.4e-4, cf=5.8e-4, m=0.0):
    """Havlena-Odeh expansion terms (rm3 per sm3): Eo, Eg, Efw."""
    i, c = pv.pvt_interp(pvt, pi), pv.pvt_interp(pvt, p)
    p = np.asarray(p, dtype=float)
    eo = (c["Bo"] - i["Bo"]) + (i["Rs"] - c["Rs"]) * c["Bg"]
    eg = i["Bo"] * (c["Bg"] / i["Bg"] - 1.0)
    efw = (1.0 + m) * i["Bo"] * (cw * swi + cf) / (1.0 - swi) * (pi - p)
    return eo, eg, efw


def ho_F(np_, gp, wp, p, pvt, bw=1.0):
    """Underground withdrawal F = Np [Bo + (Rp - Rs) Bg] + Wp Bw (rm3)."""
    np_, gp, wp = (np.asarray(a, dtype=float) for a in (np_, gp, wp))
    c = pv.pvt_interp(pvt, p)
    return np_ * (c["Bo"] + (gp / np_ - c["Rs"]) * c["Bg"]) + wp * bw


def schilthuis_integral(t_days, p, pi):
    """Cumulative integral of (pi - p) dt from t = 0 (MPa days), by trapezoids."""
    t = np.r_[0.0, np.asarray(t_days, dtype=float)]
    dp = np.r_[0.0, pi - np.asarray(p, dtype=float)]
    return np.cumsum(np.r_[0.0, 0.5 * (dp[1:] + dp[:-1]) * np.diff(t)])[1:]


def ho_prepare(hist, pvt, pi, swi=0.2, cw=4.4e-4, cf=5.8e-4, bw=1.0, m=0.0):
    """Arrays for the straight-line fits from a history table (Time_days,
    Pressure_MPa, Np_sm3, Gp_sm3, Wp_sm3). Points at or above the initial
    pressure, or with no oil produced, are dropped."""
    h = hist[(hist["Pressure_MPa"] < 0.999 * pi) & (hist["Np_sm3"] > 0)].reset_index(drop=True)
    p = h["Pressure_MPa"].to_numpy(dtype=float)
    eo, eg, efw = ho_terms(p, pvt, pi, swi, cw, cf, m)
    t = h["Time_days"].to_numpy(dtype=float)
    return {"p": p, "t": t, "np": h["Np_sm3"].to_numpy(dtype=float), "eo": eo, "eg": eg, "efw": efw,
            "F": ho_F(h["Np_sm3"], h["Gp_sm3"], h["Wp_sm3"], p, pvt, bw),
            "intdp": schilthuis_integral(h["Time_days"], hist["Pressure_MPa"], pi)[: len(hist)][
                ((hist["Pressure_MPa"] < 0.999 * pi) & (hist["Np_sm3"] > 0)).to_numpy()]}


def _r2(y, fit):
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - fit) ** 2).sum()) / ss if ss > 0 else float("nan")


def ho_depletion(F, Et, n=5000, seed=0):
    """F = N Et through the origin (no gas cap, no aquifer)."""
    F, Et = np.asarray(F, dtype=float), np.asarray(Et, dtype=float)
    N = float((F * Et).sum() / (Et ** 2).sum())
    res = F - N * Et
    se = float(np.sqrt((res ** 2).sum() / max(len(F) - 1, 1) / (Et ** 2).sum()))
    s = N + se * np.random.default_rng(seed).standard_normal(n)
    return {"N": N, "se": se, "N_samples": s[s > 0], "r2": _r2(F, N * Et), "n": len(F)}


def ho_aquifer(F, Et, intdp, n=5000, seed=0):
    """F/Et = N + C * (integral of dp dt) / Et: intercept N, slope C (rm3 per MPa day)."""
    y, x = np.asarray(F, dtype=float) / Et, np.asarray(intdp, dtype=float) / Et
    (c, N), cov = np.polyfit(x, y, 1, cov=True)
    d = _mvn([N, c], cov, n, np.random.default_rng(seed))
    ok = d[:, 0] > 0
    return {"N": float(N), "C": float(c), "N_samples": d[ok, 0], "C_samples": d[ok, 1], "r2": _r2(y, N + c * x), "n": len(y)}


def ho_gascap(F, eo_fw, eg, n=5000, seed=0):
    """F = N (Eo + Efw) + (N m) Eg: two coefficients, so N and the gas-cap ratio m."""
    X = np.column_stack([eo_fw, eg])
    F = np.asarray(F, dtype=float)
    beta = np.linalg.lstsq(X, F, rcond=None)[0]
    res = F - X @ beta
    cov = float((res ** 2).sum() / max(len(F) - 2, 1)) * np.linalg.inv(X.T @ X)
    d = _mvn(beta, cov, n, np.random.default_rng(seed))
    ok = (d[:, 0] > 0) & (d[:, 1] >= 0)
    return {"N": float(beta[0]), "m": float(beta[1] / beta[0]), "N_samples": d[ok, 0],
            "m_samples": d[ok, 1] / d[ok, 0], "r2": _r2(F, X @ beta), "n": len(F)}


def _ho_point(method, d):
    Et = d["eo"] + d["efw"]
    if method == "depletion":
        return float((d["F"] * Et).sum() / (Et ** 2).sum()), float("nan")
    if method == "aquifer":
        c, N = np.polyfit(d["intdp"] / Et, d["F"] / Et, 1)
        return float(N), float(c)
    beta = np.linalg.lstsq(np.column_stack([Et, d["eg"]]), d["F"], rcond=None)[0]
    return float(beta[0]), float(beta[1] / beta[0])


def ho_monte_carlo(hist, pvt, pi, method, p_rel=0.005, vol_rel=0.01, n=400, seed=0, **props):
    """Refit the Havlena-Odeh model many times with measurement error added to
    pressures (relative, p_rel) and cumulative volumes (vol_rel). This captures the
    error in the pressure that enters both axes, which plain regression ignores.
    method: 'depletion', 'aquifer' or 'gascap'. Returns arrays of N and the
    second parameter (C for aquifer, m for gas cap, NaN for depletion)."""
    rng = np.random.default_rng(seed)
    Ns, extra = [], []
    for _ in range(n):
        h = hist.copy()
        h["Pressure_MPa"] = h["Pressure_MPa"] * (1.0 + rng.normal(0, p_rel, len(h)))
        for c in ["Np_sm3", "Gp_sm3", "Wp_sm3"]:
            h[c] = h[c] * (1.0 + rng.normal(0, vol_rel, len(h)))
        try:
            d = ho_prepare(h, pvt, pi * (1.0 + rng.normal(0, p_rel)), **props)
            N, e = _ho_point(method, d)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        if np.isfinite(N) and N > 0:
            Ns.append(N)
            extra.append(e)
    return {"N_samples": np.array(Ns), "extra_samples": np.array(extra)}


def drive_hint(F, Et):
    """Havlena-Odeh diagnostic: F/Et against time. Flat means the assumed
    mechanism fits; a rising trend suggests extra energy (water influx or a gas cap)."""
    y = np.asarray(F, dtype=float) / np.asarray(Et, dtype=float)
    if len(y) < 5:
        return "Too few points for a drive diagnostic."
    s = float(np.polyfit(np.arange(len(y)), y / y.mean(), 1)[0]) * len(y)
    if s > 0.25:
        return "F/Et rises over time: extra energy is likely (water influx or an unmodelled gas cap)."
    if s < -0.25:
        return "F/Et falls over time: check pressures and PVT, or consider compaction effects."
    return ("F/Et is roughly flat. That fits the assumed mechanism, but it does not rule out aquifer support when "
            "water influx grows in step with the expansion terms: N and the aquifer constant are then hard to separate.")


# END OF FILE: geoflux_mb.py