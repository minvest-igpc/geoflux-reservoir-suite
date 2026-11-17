"""GeoFlux RTA-lite: rate-transient analysis of rate and flowing-pressure data.

Single-phase liquid (slightly compressible) only. Gas needs pseudo-pressure and
pseudo-time, which are NOT included. Units: pressure MPa, rate sm3/d, time days,
viscosity mPa.s (cP), compressibility 1/MPa, permeability md.

Rate-normalised pressure  RNP = (pi - pwf) / (q Bo)        [MPa d / m3]
Material balance time     tmb = Np / q                      [d]
Square-root superposition time  x = sum_j (q_j - q_{j-1}) sqrt(t - t_{j-1}) / q   (exact for linear flow
                          with varying rate and pressure; tmb is only an approximation)
Transient LINEAR flow:    RNP = m_L x + b,   A sqrt(k) = (2 / m_L) sqrt(mu / (pi phi ct))
Boundary-dominated flow:  RNP = tmb / (ct Vp) + b'          (Vp = pore volume)

Only the product A sqrt(k) is identifiable from linear flow: fracture area A and
permeability k trade off against each other (see xf_k_tradeoff).
"""
import numpy as np

KMD = 9.869233e-16                 # m2 per millidarcy
SQRT_KMD = float(np.sqrt(KMD))
SEC_PER_DAY = 86400.0


def prepare(hist, pi, bo=1.0):
    """Arrays for the diagnostics from a history table (Time_days = reporting
    time, Rate_sm3d, Pwf_MPa). Each record is a period average. Rows with no
    rate, or a flowing pressure at or above pi, are dropped."""
    t_end = hist["Time_days"].to_numpy(dtype=float)
    q, pwf = hist["Rate_sm3d"].to_numpy(dtype=float), hist["Pwf_MPa"].to_numpy(dtype=float)
    first = np.median(np.diff(t_end)) if len(t_end) > 1 else 30.4375
    d = np.diff(t_end, prepend=t_end[0] - first)
    t_start = np.cumsum(d) - d
    t_mid = t_start + d / 2.0
    np_mid = np.cumsum(q * d) - q * d / 2.0
    dq = np.diff(q, prepend=0.0)
    x_sp = np.array([float((dq[: n + 1] * np.sqrt(t_mid[n] - t_start[: n + 1])).sum() / q[n]) if q[n] > 0 else np.nan
                     for n in range(len(q))])
    ok = (q > 0) & (pwf < pi) & np.isfinite(q) & np.isfinite(pwf) & (np_mid > 0)
    q, pwf, t_mid, np_mid, x_sp = q[ok], pwf[ok], t_mid[ok], np_mid[ok], x_sp[ok]
    return {"t": t_mid, "q": q, "pwf": pwf, "np": np_mid, "tmb": np_mid / q, "rnp": (pi - pwf) / (q * bo), "x_sp": x_sp}


def loglog_slope(tmb, rnp, window=5):
    """Local slope of log RNP against log tmb (centred window). About 0.5 means
    linear flow, about 1 boundary-dominated flow, about 0.25 bilinear flow."""
    x, y = np.log(np.asarray(tmb, dtype=float)), np.log(np.asarray(rnp, dtype=float))
    out = np.full(len(x), np.nan)
    h = window // 2
    for i in range(h, len(x) - h):
        out[i] = np.polyfit(x[i - h:i + h + 1], y[i - h:i + h + 1], 1)[0]
    return out


def classify(slope):
    s = np.asarray(slope, dtype=float)
    return np.where(~np.isfinite(s), "n/a", np.where(s < 0.35, "early / bilinear-like",
                    np.where(s <= 0.65, "linear", np.where(s < 0.85, "transition", "boundary-dominated"))))


def auto_linear_window(slope, min_points=6):
    """Longest run of consecutive points whose slope is within 0.35 to 0.65.
    Returns (start, end) indices, or None if no run is long enough."""
    ok = np.isfinite(slope) & (slope >= 0.35) & (slope <= 0.65)
    best, start = None, None
    for i, v in enumerate(list(ok) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_points and (best is None or i - start > best[1] - best[0] + 1):
                best = (start, i - 1)
            start = None
    return best


def _line(x, y):
    (m, b), cov = np.polyfit(x, y, 1, cov=True)
    fit = m * x + b
    ss = float(((y - y.mean()) ** 2).sum())
    return {"m": float(m), "b": float(b), "se_m": float(np.sqrt(cov[0, 0])), "n": len(x),
            "r2": 1.0 - float(((y - fit) ** 2).sum()) / ss if ss > 0 else float("nan")}


def linear_fit(x, rnp, i0, i1):
    """Fit RNP against x over points i0..i1 (inclusive). Pass the square-root
    superposition time x_sp (exact for linear flow) or sqrt(tmb) (approximate)."""
    if i1 - i0 < 3:
        raise ValueError("Need at least 4 points in the window.")
    return _line(np.asarray(x)[i0:i1 + 1], np.asarray(rnp)[i0:i1 + 1])


def a_sqrt_k(m_user, phi, ct_per_mpa, mu_cp):
    """A sqrt(k) in m2 sqrt(md) from the sqrt-time slope (MPa d / m3 per sqrt(d)):
    A sqrt(k) = (2 / m_L) sqrt(mu / (pi phi ct)), converted from SI."""
    m_si = 1e6 * np.sqrt(SEC_PER_DAY) * m_user
    c_si = 2.0 / m_si * np.sqrt(mu_cp * 1e-3 / (np.pi * phi * ct_per_mpa / 1e6))
    return float(c_si / SQRT_KMD)


def xf_from_k(c, k_md, h_m, n_frac):
    """Fracture half-length (m) if permeability is known: A = c / sqrt(k), A = 2 xf h n."""
    return float(c / np.sqrt(k_md) / (2.0 * h_m * n_frac))


def k_from_xf(c, xf_m, h_m, n_frac):
    """Permeability (md) if fracture half-length is known."""
    return float((c / (2.0 * xf_m * h_m * n_frac)) ** 2)


def xf_k_tradeoff(c, h_m, n_frac, xf_values):
    """The consistent (xf, k) pairs: every row fits the data equally well."""
    return [{"xf_m": float(x), "k_md": k_from_xf(c, x, h_m, n_frac)} for x in xf_values]


def joint_fit(x_sp, tmb, rnp, i0, i1):
    """Fit RNP = b + a x + c tmb over points i0..i1, with x the square-root
    superposition time (exact for linear flow) and tmb exact for boundary-dominated
    flow. a is the linear-flow slope, corrected for boundary effects; c is the
    boundary slope. Also returns standard errors and the t-statistic of c."""
    if i1 - i0 < 5:
        raise ValueError("Need at least 6 points in the window.")
    x, t, y = (np.asarray(v, dtype=float)[i0:i1 + 1] for v in (x_sp, tmb, rnp))
    X = np.column_stack([np.ones_like(t), x, t])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    se = np.sqrt(np.diag(float((res ** 2).sum() / max(len(t) - 3, 1)) * np.linalg.inv(X.T @ X)))
    return {"b": float(beta[0]), "a": float(beta[1]), "c": float(beta[2]), "se_a": float(se[1]),
            "se_c": float(se[2]), "t_c": float(beta[2] / se[2]) if se[2] > 0 else float("inf"), "n": len(t)}


def boundary_detected(joint, threshold=5.0):
    """True when the boundary slope is positive and clearly non-zero (t above 5;
    a threshold of 3 gave about 10% false alarms on linear-only data in testing)."""
    return bool(joint["c"] > 0 and joint["t_c"] > threshold)


def bdf_fit(tmb, rnp, i0, i1, ct_per_mpa, sw=0.0, bo=1.0, x_sp=None):
    """Boundary-dominated fit: c from joint_fit gives the pore volume
    Vp = 1 / (ct c) in reservoir m3, then hydrocarbon in place (sm3). The linear term
    uses square-root superposition time when x_sp is given (else sqrt(tmb)).
    Needs a long, clearly late-time window."""
    lin = np.sqrt(np.asarray(tmb, dtype=float)) if x_sp is None else x_sp
    j = joint_fit(lin, tmb, rnp, i0, i1)
    if j["c"] <= 0:
        raise ValueError("The boundary-dominated slope is not positive: boundary-dominated flow is not visible.")
    vp = 1.0 / (ct_per_mpa * j["c"])
    return {**j, "pore_volume_m3": float(vp), "hcpv_m3": float(vp * (1.0 - sw)),
            "in_place_sm3": float(vp * (1.0 - sw) / bo), "vp_rel_se": j["se_c"] / j["c"]}


def a_sqrt_k_mc(hist, pi, bo, window, phi, ct_per_mpa, mu_cp, rel_rate=0.03, rel_pwf=0.015,
                rel_phi=0.10, rel_ct=0.15, rel_mu=0.05, n=300, seed=0, joint=False):
    """Monte Carlo of A sqrt(k): rates and flowing pressures are perturbed by their
    measurement errors (relative sd) and the straight line is refitted over the same
    window of points, then porosity, compressibility and viscosity are drawn with
    lognormal uncertainty. Regression error bars alone are too narrow because the
    noise enters both axes. With joint=True the linear slope comes from the joint
    linear + boundary fit. A sqrt(k) scales as sqrt(mu / (phi ct))."""
    rng = np.random.default_rng(seed)
    n_pts = len(prepare(hist, pi, bo)["tmb"])
    ms = []
    for _ in range(n):
        h = hist.copy()
        h["Rate_sm3d"] = h["Rate_sm3d"] * np.exp(rng.normal(0, rel_rate, len(h)))
        h["Pwf_MPa"] = h["Pwf_MPa"] * (1.0 + rng.normal(0, rel_pwf, len(h)))
        d = prepare(h, pi * (1.0 + rng.normal(0, rel_pwf / 3.0)), bo)
        if len(d["tmb"]) != n_pts:
            continue
        m = joint_fit(d["x_sp"], d["tmb"], d["rnp"], *window)["a"] if joint else linear_fit(d["x_sp"], d["rnp"], *window)["m"]
        if m > 0:
            ms.append(m)
    ms = np.array(ms)
    f = lambda rel: np.exp(rng.normal(0.0, rel, len(ms)))  # noqa: E731
    return np.array([a_sqrt_k(m, phi * x, ct_per_mpa * y, mu_cp * z)
                     for m, x, y, z in zip(ms, f(rel_phi), f(rel_ct), f(rel_mu))])


# END OF FILE: geoflux_rta.py