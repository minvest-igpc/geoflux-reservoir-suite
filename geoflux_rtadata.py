"""GeoFlux RTA data: loader and a synthetic well with known answers."""
import numpy as np
import pandas as pd

import geoflux_mine as mine
import geoflux_pvt as pv
import geoflux_rta as rta

BBL_TO_M3 = 0.158987294928
_ALIASES = {"date": "Date", "time": "Date", "time_days": "Time_days", "days": "Time_days",
            "rate_sm3d": "Rate_sm3d", "q_sm3d": "Rate_sm3d", "oil_rate_sm3d": "Rate_sm3d", "rate_bbld": "Rate_bbld",
            "oil_rate_bbld": "Rate_bbld", "pwf_mpa": "Pwf_MPa", "pwf": "Pwf_MPa", "bhp_mpa": "Pwf_MPa",
            "flowing_pressure_mpa": "Pwf_MPa", "pwf_psi": "Pwf_psi", "bhp_psi": "Pwf_psi"}


def load_rta_history(file):
    """Period-average rate and flowing pressure. Needs a Date or Time_days, a
    rate (Rate_sm3d or Rate_bbld) and Pwf_MPa (or Pwf_psi)."""
    warnings = []
    try:
        df = mine._read(file)
    except Exception as exc:
        return None, f"Could not read the file: {exc}", warnings
    df = mine._normalise(df, _ALIASES)
    if "Rate_sm3d" not in df.columns and "Rate_bbld" in df.columns:
        df["Rate_sm3d"] = pd.to_numeric(df["Rate_bbld"], errors="coerce") * BBL_TO_M3
        warnings.append("Rate converted from barrels per day to sm3 per day.")
    if "Pwf_MPa" not in df.columns and "Pwf_psi" in df.columns:
        df["Pwf_MPa"] = pd.to_numeric(df["Pwf_psi"], errors="coerce") / pv.MPA_TO_PSI
        warnings.append("Flowing pressure converted from psi to MPa.")
    missing = [c for c in ["Rate_sm3d", "Pwf_MPa"] if c not in df.columns]
    if missing:
        return None, f"Missing columns: {missing}. Found: {list(df.columns)}", warnings
    if "Time_days" not in df.columns:
        if "Date" not in df.columns:
            return None, "Need a Date or Time_days column.", warnings
        d = mine._parse_time(df["Date"], warnings)
        df["Time_days"] = (d - d.min()).dt.total_seconds() / 86400.0
        if len(df):
            df["Time_days"] = df["Time_days"] + (df["Time_days"].diff().median() if len(df) > 1 else 30.4375)
    for c in ["Time_days", "Rate_sm3d", "Pwf_MPa"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Time_days", "Rate_sm3d", "Pwf_MPa"])
    df = df[df["Rate_sm3d"] > 0].sort_values("Time_days").reset_index(drop=True)
    if len(df) < 8:
        return None, "Need at least 8 valid rows.", warnings
    return df[["Time_days", "Rate_sm3d", "Pwf_MPa"]], None, warnings


def make_synthetic_rta(seed=4, c_true=12000.0, phi=0.08, ct=2e-3, mu=1.0, bo=1.2, pi=40.0,
                       months=36, bdf_at_days=None, noise=0.02):
    """Synthetic well: a declining flowing-pressure schedule produces rates through
    exact superposition of the linear-flow response (optionally with a boundary
    term that dominates after bdf_at_days). Daily steps are averaged to 30-day
    records, then noise is added. c_true is A sqrt(k) in m2 sqrt(md).
    Returns (history, truth)."""
    rng = np.random.default_rng(seed)
    days = int(months * 30)
    c_si = c_true * rta.SQRT_KMD
    ct_pa = ct / 1e6
    m_si = 2.0 / c_si * np.sqrt(mu * 1e-3 / (np.pi * phi * ct_pa))
    vp = inv = None
    if bdf_at_days:
        vp = float(np.sqrt(bdf_at_days * 86400.0) / (m_si * ct_pa))
        inv = 1.0 / (ct_pa * vp)

    def g(ts):
        return m_si * np.sqrt(ts) + (inv * ts if inv else 0.0)

    t_end = np.arange(1, days + 1) * 86400.0
    t_start = t_end - 86400.0
    pwf = 10.0 + 25.0 * np.exp(-np.arange(1, days + 1) / 90.0)
    dp = (pi - pwf) * 1e6
    q, dq = np.zeros(days), np.zeros(days)
    for n in range(days):
        prev = float(dq[:n] @ g(t_end[n] - t_start[:n])) if n else 0.0
        dq[n] = (dp[n] - prev) / g(t_end[n] - t_start[n])
        q[n] = (q[n - 1] if n else 0.0) + dq[n]
    blocks = days // 30
    q_sm3d = q[: blocks * 30].reshape(blocks, 30).mean(axis=1) * 86400.0 / bo
    p_blk = pwf[: blocks * 30].reshape(blocks, 30).mean(axis=1)
    hist = pd.DataFrame({"Time_days": 30.0 * np.arange(1, blocks + 1),
                         "Rate_sm3d": q_sm3d * np.exp(rng.normal(0, noise, blocks)),
                         "Pwf_MPa": p_blk * (1.0 + rng.normal(0, noise / 2.0, blocks))})
    return hist, {"c": c_true, "phi": phi, "ct": ct, "mu": mu, "bo": bo, "pi": pi, "vp_m3": vp,
                  "in_place_sm3": vp / bo if vp else None, "bdf_at_days": bdf_at_days}


# END OF FILE: geoflux_rtadata.py