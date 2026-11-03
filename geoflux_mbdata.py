"""GeoFlux material balance data: loaders and synthetic reservoirs with known answers."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

import geoflux_mb as mb
import geoflux_mine as mine
import geoflux_pvt as pv

_HIST_ALIASES = {"date": "Date", "time": "Date", "time_days": "Time_days", "days": "Time_days",
                 "pressure_mpa": "Pressure_MPa", "p_mpa": "Pressure_MPa", "pressure": "Pressure_MPa",
                 "pressure_psi": "Pressure_psi", "np": "Np_sm3", "np_sm3": "Np_sm3", "cum_oil_sm3": "Np_sm3",
                 "gp": "Gp_sm3", "gp_sm3": "Gp_sm3", "cum_gas_sm3": "Gp_sm3",
                 "wp": "Wp_sm3", "wp_sm3": "Wp_sm3", "cum_water_sm3": "Wp_sm3"}
_PVT_ALIASES = {"pressure_mpa": "Pressure_MPa", "pressure": "Pressure_MPa", "p_mpa": "Pressure_MPa",
                "bo": "Bo", "rs": "Rs", "bg": "Bg"}


def load_mb_history(file):
    """Pressure and cumulative production history. Needs Pressure_MPa (or
    Pressure_psi) and a Date or Time_days; Np_sm3, Gp_sm3, Wp_sm3 as available."""
    warnings = []
    try:
        df = mine._read(file)
    except Exception as exc:
        return None, f"Could not read the file: {exc}", warnings
    df = mine._normalise(df, _HIST_ALIASES)
    if "Pressure_MPa" not in df.columns and "Pressure_psi" in df.columns:
        df["Pressure_MPa"] = pd.to_numeric(df["Pressure_psi"], errors="coerce") / pv.MPA_TO_PSI
        warnings.append("Pressure converted from psi to MPa.")
    if "Pressure_MPa" not in df.columns:
        return None, f"Need a Pressure_MPa column. Found: {list(df.columns)}", warnings
    if "Time_days" not in df.columns:
        if "Date" not in df.columns:
            return None, "Need a Date or Time_days column.", warnings
        d = mine._parse_time(df["Date"], warnings)
        df["Time_days"] = (d - d.min()).dt.total_seconds() / 86400.0
    for c in ["Time_days", "Pressure_MPa", "Np_sm3", "Gp_sm3", "Wp_sm3"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Np_sm3", "Gp_sm3", "Wp_sm3"]:
        if c not in df.columns:
            df[c] = 0.0
    df = df.dropna(subset=["Time_days", "Pressure_MPa"]).fillna({"Np_sm3": 0.0, "Gp_sm3": 0.0, "Wp_sm3": 0.0})
    df = df[df["Pressure_MPa"] > 0].sort_values("Time_days").reset_index(drop=True)
    if len(df) < 4:
        return None, "Need at least 4 valid rows.", warnings
    return df[["Time_days", "Pressure_MPa", "Np_sm3", "Gp_sm3", "Wp_sm3"]], None, warnings


def load_pvt(file):
    """PVT table: Pressure_MPa, Bo, Rs, Bg (rm3/sm3, sm3/sm3)."""
    try:
        df = mine._normalise(mine._read(file), _PVT_ALIASES)
    except Exception as exc:
        return None, f"Could not read the PVT file: {exc}"
    missing = [c for c in ["Pressure_MPa", "Bo", "Rs", "Bg"] if c not in df.columns]
    if missing:
        return None, f"PVT file missing columns: {missing}. Found: {list(df.columns)}"
    df = df[["Pressure_MPa", "Bo", "Rs", "Bg"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 4:
        return None, "Need at least 4 valid PVT rows."
    return df.sort_values("Pressure_MPa").reset_index(drop=True), None


def make_synthetic_oil_mb(seed=1, N=50e6, m=0.0, C=0.0, noise=0.002, n_pts=14, pi=30.0,
                          swi=0.2, cw=4.4e-4, cf=5.8e-4):
    """Oil reservoir with known N, gas-cap ratio m and aquifer constant C. The
    production history is generated so the material balance holds exactly, then
    pressure noise is added. Returns (history, pvt, truth)."""
    rng = np.random.default_rng(seed)
    pvt = pv.synthetic_oil_pvt()
    p = np.linspace(pi, 14.0, n_pts + 1)[1:]
    t = 180.0 * np.arange(1, n_pts + 1)
    eo, eg, efw = mb.ho_terms(p, pvt, pi, swi, cw, cf, m)
    F = N * (eo + m * eg + efw) + C * mb.schilthuis_integral(t, p, pi)
    c = pv.pvt_interp(pvt, p)
    rp = c["Rs"] + 40.0 * (pi - p) / pi * 3.0                    # producing GOR above solution GOR
    np_ = F / (c["Bo"] + (rp - c["Rs"]) * c["Bg"])
    hist = pd.DataFrame({"Time_days": t, "Pressure_MPa": p * (1.0 + rng.normal(0, noise, n_pts)),
                         "Np_sm3": np_, "Gp_sm3": np_ * rp, "Wp_sm3": 0.0})
    return hist, pvt, {"N": N, "m": m, "C": C, "pi": pi, "swi": swi, "cw": cw, "cf": cf}


def make_synthetic_gas_mb(seed=2, G=10e9, pi=35.0, t_c=100.0, gamma=0.7, noise=0.01, n_pts=12, curvature=0.0):
    """Gas reservoir with known G. curvature > 0 bends the P/Z line, as water
    influx or compaction would. Returns (history, truth)."""
    rng = np.random.default_rng(seed)
    pz_i = pi / float(pv.z_papay(pi, t_c, gamma))
    f = np.linspace(0.05, 0.55, n_pts)
    pz = pz_i * (1.0 - f) ** (1.0 + curvature)
    p = np.array([brentq(lambda x, target=target: x / float(pv.z_papay(x, t_c, gamma)) - target, 0.5, 60.0) for target in pz])
    hist = pd.DataFrame({"Time_days": 120.0 * np.arange(1, n_pts + 1), "Pressure_MPa": p * (1.0 + rng.normal(0, noise, n_pts)),
                         "Gp_sm3": G * f})
    return hist, {"G": G, "pi": pi, "t_c": t_c, "gamma": gamma}


# END OF FILE: geoflux_mbdata.py