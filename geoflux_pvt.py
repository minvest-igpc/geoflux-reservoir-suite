"""GeoFlux PVT helpers (SI units): gas z-factor, gas formation volume factor,
and a synthetic oil PVT table for testing.

Units: pressure MPa, temperature degC, volumes sm3 (standard) and rm3 (reservoir).
The z-factor here is the Papay correlation with Standing pseudo-criticals for dry
gas: an approximation. Use laboratory or equation-of-state z-factors when you have them.
"""
import numpy as np
import pandas as pd

P_SC_KPA = 101.325
T_SC_K = 288.15
MPA_TO_PSI = 145.0377


def z_papay(p_mpa, t_c, gamma_g):
    """Approximate gas z-factor (Papay). Valid roughly for pseudo-reduced
    pressure below 15 and pseudo-reduced temperature of 1.2 to 2.4."""
    g = float(gamma_g)
    t_pc = 168.0 + 325.0 * g - 12.5 * g * g              # degrees Rankine
    p_pc = 677.0 + 15.0 * g - 37.5 * g * g               # psia
    t_pr = (t_c * 9.0 / 5.0 + 32.0 + 459.67) / t_pc
    p_pr = np.asarray(p_mpa, dtype=float) * MPA_TO_PSI / p_pc
    z = 1.0 - 3.52 * p_pr / 10.0 ** (0.9813 * t_pr) + 0.274 * p_pr ** 2 / 10.0 ** (0.8157 * t_pr)
    return np.clip(z, 0.2, 1.6)


def bg_rm3_sm3(p_mpa, t_c, z):
    """Gas formation volume factor: Bg = (p_sc / T_sc) * z * T / p."""
    return (P_SC_KPA / T_SC_K) * np.asarray(z, dtype=float) * (t_c + 273.15) / (np.asarray(p_mpa, dtype=float) * 1000.0)


def pvt_interp(pvt, p_mpa):
    """Interpolate a PVT table (columns Pressure_MPa, Bo, Rs, Bg) at pressure(s)."""
    p = np.asarray(p_mpa, dtype=float)
    x = pvt["Pressure_MPa"].to_numpy(dtype=float)
    order = np.argsort(x)
    return {c: np.interp(p, x[order], pvt[c].to_numpy(dtype=float)[order]) for c in ["Bo", "Rs", "Bg"]}


def synthetic_oil_pvt(p_bubble=25.0, rs_bubble=120.0, bo_bubble=1.38, c_o=1.5e-3, t_c=100.0, gamma_g=0.7):
    """Black-oil table: Rs and Bo fall linearly below the bubble point; above it
    Bo shrinks with oil compressibility. Bg from the Papay z-factor."""
    p = np.linspace(4.0, 40.0, 73)
    rs = np.where(p <= p_bubble, rs_bubble * p / p_bubble, rs_bubble)
    bo = np.where(p <= p_bubble, 1.0 + (bo_bubble - 1.0) * rs / rs_bubble, bo_bubble * np.exp(-c_o * (p - p_bubble)))
    return pd.DataFrame({"Pressure_MPa": p, "Bo": bo, "Rs": rs, "Bg": bg_rm3_sm3(p, t_c, z_papay(p, t_c, gamma_g))})


# END OF FILE: geoflux_pvt.py