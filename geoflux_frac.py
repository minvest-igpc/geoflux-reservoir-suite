"""GeoFlux oil & gas: stage-linked induced seismicity and SRV.

Seismogenic index, McGarr bound, traffic light, stimulated rock volume and
well/fault geometry. Standard documented methods; nothing is site-calibrated.
"""
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def assign_stages(times, stages, lag_min=60.0):
    """Label each event with the stage being pumped, or pumped within the last
    lag_min minutes. Events outside every window get ''. If windows overlap the
    most recently started stage wins."""
    t = pd.Series(times).to_numpy("datetime64[ns]")
    starts = stages["Start"].to_numpy("datetime64[ns]")
    ends = stages["End"].to_numpy("datetime64[ns]") + np.timedelta64(int(lag_min * 60), "s")
    labels = stages["Stage"].to_numpy(dtype=object)
    idx = np.searchsorted(starts, t, side="right") - 1
    ic = np.clip(idx, 0, None)
    within = (idx >= 0) & (t <= ends[ic])
    return np.where(within, labels[ic], "").astype(object)


def mw_to_m0(mw):
    """Seismic moment (N m) from moment magnitude: Mw = (2/3)(log10 M0 - 9.1)."""
    return 10.0 ** (1.5 * np.asarray(mw, dtype=float) + 9.1)


def m0_to_mw(m0):
    return (np.log10(np.asarray(m0, dtype=float)) - 9.1) / 1.5


def mcgarr_bound(volume_m3, shear_modulus_pa=30e9):
    """McGarr (2014) upper bound on the largest induced moment for a given net
    injected volume: M0_max = G * dV. A physical-limit argument, not a guarantee;
    magnitudes in the catalogue are treated as moment magnitudes."""
    m0 = shear_modulus_pa * float(volume_m3)
    return {"M0_max": m0, "Mw_max": float(m0_to_mw(m0))}


def seismogenic_index(n_events, volume_m3, b, mc):
    """Shapiro seismogenic index: Sigma = log10(N(>=Mc)) - log10(V) + b*Mc,
    so that N(>=M) = V * 10^(Sigma - b*M). Higher Sigma = more seismogenic."""
    if n_events <= 0 or volume_m3 <= 0:
        return np.nan
    return float(np.log10(n_events) - np.log10(volume_m3) + b * mc)


def exceedance_from_index(sigma, b, volume_m3, m_star, mc):
    """P(at least one event >= m_star) for injecting volume_m3, from the
    seismogenic index (Poisson assumption)."""
    n = volume_m3 * 10.0 ** (sigma - b * max(float(m_star), float(mc)))
    return float(1.0 - np.exp(-n))


def traffic_light(max_magnitude, amber, red):
    if not np.isfinite(max_magnitude):
        return "Green"
    return "Red" if max_magnitude >= red else "Amber" if max_magnitude >= amber else "Green"


def stage_summary(cat, stages, mc, b, shear_modulus_pa=30e9, amber=1.0, red=2.0):
    rows = []
    for _, s in stages.iterrows():
        sel = cat[cat["Stage"] == s["Stage"]]
        n_mc = int((sel["Magnitude"] >= mc - 1e-9).sum())
        v = float(s["Volume_m3"])
        mmax = float(sel["Magnitude"].max()) if len(sel) else np.nan
        m0 = float(mw_to_m0(sel["Magnitude"]).sum()) if len(sel) else 0.0
        bound = mcgarr_bound(v, shear_modulus_pa)
        max_m0 = float(mw_to_m0(mmax)) if len(sel) else 0.0
        rows.append({
            "Stage": s["Stage"], "Volume_m3": v, "Events": len(sel), "Events_above_Mc": n_mc,
            "Mmax": mmax, "Sigma": seismogenic_index(n_mc, v, b, mc),
            "Sigma_err": 1.0 / (np.log(10) * np.sqrt(n_mc)) if n_mc > 0 else np.nan,
            "Cum_moment_Nm": m0, "McGarr_Mw": bound["Mw_max"],
            "McGarr_utilisation": max_m0 / bound["M0_max"] if len(sel) else 0.0,
            "Status": traffic_light(mmax, amber, red),
        })
    return pd.DataFrame(rows)


def srv_hull(xyz):
    """Convex-hull volume (m3): an upper bound, sensitive to outlier events."""
    xyz = np.asarray(xyz, dtype=float)
    if len(xyz) < 4:
        return None
    try:
        return float(ConvexHull(xyz).volume)
    except Exception:
        return None


def srv_voxel(xyz, voxel, min_events=2):
    """Volume of voxels holding at least min_events events. Quantised: edge
    voxels make it overestimate by up to about one voxel width per dimension.
    Do not use a voxel smaller than the location error."""
    xyz = np.asarray(xyz, dtype=float)
    if len(xyz) == 0:
        return 0.0
    idx = np.floor(xyz / voxel).astype(np.int64)
    _, counts = np.unique(idx, axis=0, return_counts=True)
    return float((counts >= min_events).sum() * voxel ** 3)


def srv_sensitivity(xyz, voxel, min_events=2, voxel_factors=(0.75, 1.0, 1.5),
                    min_events_list=None):
    """Voxel SRV over a neighbourhood of the chosen settings: voxel size at
    -25%, 0%, +50% and the minimum-event threshold at -1, 0, +1. This shows how
    much of the answer is a method choice. It is a sensitivity range, NOT a
    statistical confidence interval (a bootstrap was rejected: duplicated events
    would wrongly satisfy the minimum-event rule)."""
    if min_events_list is None:
        min_events_list = sorted({max(1, min_events - 1), min_events, min_events + 1})
    rows = []
    for f in voxel_factors:
        for m in min_events_list:
            rows.append({"Voxel_m": float(voxel * f), "Min_events": int(m),
                         "SRV_m3": srv_voxel(xyz, voxel * f, m)})
    return pd.DataFrame(rows)


def densify(poly_df, step=5.0):
    """Points every ~step metres along each polyline (for min-distance work)."""
    pts = []
    for _, g in poly_df.groupby("Name", sort=False):
        V = g[["X", "Y", "Z"]].to_numpy(dtype=float)
        if len(V) == 1:
            pts.append(V)
            continue
        for a, b in zip(V[:-1], V[1:]):
            k = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1) + 1
            pts.append(a + np.linspace(0, 1, k)[:, None] * (b - a))
    return pd.DataFrame(np.vstack(pts), columns=["X", "Y", "Z"])


# END OF FILE: geoflux_frac.py