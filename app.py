# geoflux/app.py - Subsurface microseismic analytics PROTOTYPE (Phase 0 cleanup)
#
# STATUS: ILLUSTRATIVE PROTOTYPE. Outputs are not validated. Do not use for
# safety, reserves booking, regulatory, or investment decisions.
# See the CHANGELOG comment at the bottom of this file for what changed in Phase 0.
 
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import ConvexHull
 
st.set_page_config(page_title="GeoFlux Reservoir Suite (prototype)", layout="wide")
 
# --------------------------------------------------------------------------
# Constants and configuration
# --------------------------------------------------------------------------
REQUIRED_COLUMNS = ["Easting_m", "Northing_m", "Depth_m", "Mechanism", "Energy_J"]
VALID_MECHANISMS = ["Shear", "Tensile", "Closure"]
RENAME_MAP = {
    "Energy": "Energy_J", "Energy_Joules": "Energy_J",
    "X": "Easting_m", "Y": "Northing_m", "Z": "Depth_m",
    "Mechanism_Type": "Mechanism", "Source_Mechanism": "Mechanism",
}
 
# Regimes are looked up by exact key, never by substring.
# ("Ultra-Deep" contains the text "Deep", which broke the original substring checks.)
REGIMES = {
    "Intermediate (1,000m - 3,000m)": {"base_depth_m": 2000.0, "water_exponent": 0.75},
    "Deep (3,000m - 4,500m)":         {"base_depth_m": 3800.0, "water_exponent": 0.35},
    "Ultra-Deep (>4,500m)":           {"base_depth_m": 5000.0, "water_exponent": 0.50},
}
 
SYNTH_MC = -2.0               # completeness magnitude of the synthetic catalogue
HORIZON_DAYS = 3652           # 10-year forecast horizon
# PLACEHOLDER forecast constants. These are uncalibrated and exist only so the
# demo produces a curve. Replace with real reservoir inputs (Phase 4).
QI_BASE, QI_PER_FCD, QI_MIN, QI_MAX = 1000.0, 55.0, 800.0, 6500.0
FCD_PROXY_SCALE = 55.0        # PLACEHOLDER scaling of the conductivity proxy
 
 
# --------------------------------------------------------------------------
# Seismology helpers
# --------------------------------------------------------------------------
def energy_to_magnitude(energy_j):
    """Gutenberg-Richter energy-magnitude relation: log10(E[J]) = 1.5*M + 4.8.
    Assumes Energy_J is radiated seismic energy in joules."""
    return (np.log10(np.maximum(np.asarray(energy_j, dtype=float), 1e-12)) - 4.8) / 1.5
 
 
def magnitude_to_energy(m):
    return 10.0 ** (1.5 * np.asarray(m, dtype=float) + 4.8)
 
 
def estimate_b_value(magnitudes, hist_bin=0.1, mc_correction=0.2, min_events=30):
    """Aki (1965) maximum-likelihood b-value above an estimated completeness
    magnitude Mc.
 
        b = log10(e) / (mean(M) - Mc),   sigma_b = b / sqrt(N)
 
    Mc is estimated with the maximum-curvature method (modal histogram bin)
    plus a conservative correction. Magnitudes are treated as continuous
    (they are computed from energies, not rounded). Returns None when there are
    too few events above Mc for a meaningful estimate.
    """
    m = np.asarray(magnitudes, dtype=float)
    m = m[np.isfinite(m)]
    if m.size < min_events:
        return None
    edges = np.arange(np.floor(m.min() / hist_bin) * hist_bin,
                      m.max() + 2 * hist_bin, hist_bin)
    counts, edges = np.histogram(m, bins=edges)
    mc = float(edges[int(np.argmax(counts))] + hist_bin / 2.0 + mc_correction)
    sel = m[m >= mc]
    if sel.size < min_events or sel.mean() <= mc:
        return None
    b = float(np.log10(np.e) / (sel.mean() - mc))
    return {"b": b, "sigma": b / np.sqrt(sel.size), "mc": mc, "n_used": int(sel.size)}
 
 
# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_uploaded(file):
    """Parse and validate an uploaded catalogue.
    Returns (dataframe or None, error message or None, list of warnings)."""
    warnings = []
    try:
        df = pd.read_csv(file, sep=None, engine="python")
    except Exception as exc:
        return None, f"Could not read the file: {exc}", warnings
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, (f"Missing required columns: {missing}. "
                      f"Columns found: {list(df.columns)}"), warnings
    df = df[REQUIRED_COLUMNS].copy()
    df["Mechanism"] = df["Mechanism"].astype(str).str.strip().str.title()
    for col in ["Easting_m", "Northing_m", "Depth_m", "Energy_J"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n0 = len(df)
    df = df.dropna()
    if len(df) < n0:
        warnings.append(f"Dropped {n0 - len(df)} rows with missing or non-numeric values.")
    bad = ~df["Mechanism"].isin(VALID_MECHANISMS)
    if bad.any():
        warnings.append(f"Dropped {int(bad.sum())} rows with unrecognised Mechanism "
                        f"(allowed: {VALID_MECHANISMS}).")
        df = df[~bad]
    nonpos = df["Energy_J"] <= 0
    if nonpos.any():
        warnings.append(f"Dropped {int(nonpos.sum())} rows with non-positive Energy_J.")
        df = df[~nonpos]
    if len(df) == 0:
        return None, "No valid events remain after validation.", warnings
    return df.reset_index(drop=True), None, warnings
 
 
def generate_synthetic_catalogue(n, regime_cfg, spread, tensile_pct,
                                 noise_amp, b_true, seed=42):
    """Synthetic catalogue for demonstration only. Magnitudes follow a
    Gutenberg-Richter law with b = b_true above SYNTH_MC, so the b-value
    estimator can be sanity-checked against a known answer."""
    rng = np.random.default_rng(seed)
    noise_ratio = noise_amp / 250.0
    p_shear = np.clip(1.0 - tensile_pct / 100.0 - 0.05 - noise_ratio, 0.01, 1.0)
    p_closure = np.clip(0.05 + noise_ratio, 0.01, 0.95)
    p_tensile = np.clip(1.0 - p_shear - p_closure, 0.01, 1.0)
    p = np.array([p_shear, p_tensile, p_closure])
    p = p / p.sum()
    mags = SYNTH_MC + rng.exponential(1.0 / (b_true * np.log(10.0)), n)
    return pd.DataFrame({
        "Easting_m": rng.normal(500.0, spread * 4.0, n),
        "Northing_m": rng.normal(1200.0, spread * 2.0, n),
        "Depth_m": rng.normal(regime_cfg["base_depth_m"], spread * 3.0, n),  # positive down
        "Mechanism": rng.choice(VALID_MECHANISMS, size=n, p=p),
        "Energy_J": magnitude_to_energy(mags),
    })
 
 
# --------------------------------------------------------------------------
# Analytics engine
# --------------------------------------------------------------------------
class SubsurfaceAnalyticsEngine:
    def __init__(self, df, regime_cfg):
        self.data = df
        self.regime = regime_cfg
 
    def apply_filters(self, workings_easting, buffer_m, drop_closure):
        """(1) Remove events within buffer_m of a planar working at a given
        Easting (a 1-D approximation; real 3-D distance to mine geometry is
        future work). (2) Optionally remove events LABELLED 'Closure'."""
        df = self.data
        after_buffer = df[(df["Easting_m"] - workings_easting).abs() > buffer_m]
        after_label = (after_buffer[after_buffer["Mechanism"] != "Closure"]
                       if drop_closure else after_buffer)
        info = {"removed_by_buffer": len(df) - len(after_buffer),
                "removed_by_label": len(after_buffer) - len(after_label)}
        return after_label.copy(), info
 
    def stress_estimate(self, tensile_pct):
        """Empirical gradient-based ESTIMATE in MPa. This is NOT a stress
        inversion: it is not derived from focal mechanisms."""
        depth_m = float(np.median(np.abs(self.data["Depth_m"])))
        sv = depth_m * 0.023
        shmin = sv * (0.6 + tensile_pct / 250.0)
        shmax = shmin * (1.2 + depth_m / 15000.0)
        total_e = float(self.data["Energy_J"].sum())
        closure_e = float(self.data.loc[self.data["Mechanism"] == "Closure", "Energy_J"].sum())
        return {"depth_m": depth_m, "Sv": sv, "Shmin": shmin, "Shmax": shmax,
                "closure_energy_fraction": closure_e / total_e if total_e > 0 else 0.0}
 
    def geomechanics(self, df_clean, fluid_eff, srv_scale):
        xyz = df_clean[["Easting_m", "Northing_m", "Depth_m"]].to_numpy(dtype=float)
        hull_vol, hull_ok = 0.0, False
        if len(xyz) >= 4:
            try:
                hull_vol, hull_ok = float(ConvexHull(xyz).volume), True
            except Exception:  # degenerate geometry (coplanar, collinear, ...)
                pass
        total_e = float(df_clean["Energy_J"].sum())
        tensile_e = float(df_clean.loc[df_clean["Mechanism"] == "Tensile", "Energy_J"].sum())
        tensile_frac = tensile_e / total_e if total_e > 0 else 0.0
        mags = energy_to_magnitude(df_clean["Energy_J"].to_numpy())
        return {
            "hull_ok": hull_ok,
            "hull_volume": hull_vol,
            "srv": hull_vol * srv_scale,
            # Relative proxy, NOT dimensionless Fcd and NOT in mD-ft.
            "conductivity_index": tensile_frac * FCD_PROXY_SCALE * (fluid_eff / 50.0),
            "magnitudes": mags,
            "b": estimate_b_value(mags),
        }
 
    def forecast(self, conductivity_index, noise, tensile_pct, rate_limit):
        """PLACEHOLDER decline curve (uncalibrated). Fixed vs original:
        - SRV no longer steepens the decline,
        - hyperbolic exponent n >= 0.5 so Arps b = 1/n <= 2,
        - the rate cutoff handles the 'never reaches the limit' case correctly."""
        t = np.arange(1, HORIZON_DAYS + 1)
        qi = float(np.clip(QI_BASE + conductivity_index * QI_PER_FCD, QI_MIN, QI_MAX))
        di = float(np.clip(0.45 - tensile_pct / 300.0, 0.05, 0.40))   # nominal, per day
        n = float(np.clip(0.40 + noise / 200.0, 0.5, 0.95))
        q = qi / (1.0 + di * t) ** n
        q_water = (t ** self.regime["water_exponent"]) * (q * 0.04)
        below = np.flatnonzero(q < rate_limit)
        capped = below.size == 0
        cutoff = len(t) if capped else int(below[0])
        return {"timeline": t / 365.25, "rate": q, "water": q_water,
                "cum_boe": float(q[:cutoff].sum()), "life_years": round(cutoff / 365.25, 2),
                "horizon_capped": capped, "qi": qi}
 
 
# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.header("Reservoir Optimization Console")
env = st.sidebar.selectbox("Target Depth Envelope", list(REGIMES.keys()))
 
st.sidebar.subheader("Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload event catalogue (.csv/.txt)", type=["csv", "txt"],
    help="Required columns: Easting_m, Northing_m, Depth_m, Mechanism "
         "(Shear/Tensile/Closure), Energy_J (joules). Depth in metres.")
with st.sidebar.expander("Synthetic data controls (ignored if a file is uploaded)"):
    count = st.slider("Event count", 50, 1000, 300, 50)
    noise = st.slider("Rock Mass Anisotropy (spatial spread)", 5, 60, 25, 5)
    tensile_pct = st.slider("Focal Mechanism Tensile Ratio (%)", 5, 95, 35, 5)
    noise_sim_amplitude = st.slider("Simulated machinery-noise fraction", 0, 100, 30, 10)
    b_true = st.slider("True b-value (synthetic truth)", 0.6, 1.6, 1.0, 0.1)
 
st.sidebar.subheader("Filters")
workings_easting = st.sidebar.number_input(
    "Mine working position: Easting (m)", value=500.0, step=10.0,
    help="Planar approximation of a mine working at this Easting.")
spatial_buffer = st.sidebar.slider("Distance buffer around working (m)", 0.0, 50.0, 15.0, 5.0)
drop_closure = st.sidebar.toggle(
    "Drop events labelled 'Closure' (label filter only, no waveform analysis)", value=True)
 
st.sidebar.subheader("Assumptions")
proppant_eff = st.sidebar.slider("Fluid proppant carrying efficiency (%)", 10, 100, 75, 5)
srv_scale = st.sidebar.number_input(
    "SRV scaling factor (user assumption)", min_value=0.1, max_value=20.0, value=1.0, step=0.1,
    help="1.0 = raw convex-hull volume of the filtered event cloud.")
rate_limit = st.sidebar.slider("Economic rate limit (BOE/day)", 5.0, 120.0, 25.0, 5.0)
 
# --------------------------------------------------------------------------
# Load data (no silent fallback to synthetic data)
# --------------------------------------------------------------------------
regime_cfg = REGIMES[env]
if uploaded_file is not None:
    raw_df, load_error, load_warnings = load_uploaded(uploaded_file)
    if load_error:
        st.error(f"Upload failed: {load_error}")
        st.stop()
    is_synthetic = False
    source_label = f"UPLOADED FILE: {uploaded_file.name} ({len(raw_df):,} valid events)"
else:
    load_warnings = []
    raw_df = generate_synthetic_catalogue(count, regime_cfg, noise, tensile_pct,
                                          noise_sim_amplitude, b_true)
    is_synthetic = True
    source_label = "SYNTHETIC DEMONSTRATION DATA (randomly generated, not a real site)"
 
engine = SubsurfaceAnalyticsEngine(raw_df, regime_cfg)
df_clean, filter_info = engine.apply_filters(workings_easting, spatial_buffer, drop_closure)
if len(df_clean) == 0:
    st.error("All events were removed by the filters. Reduce the buffer or disable the label filter.")
    st.stop()
 
stress = engine.stress_estimate(tensile_pct)
geo = engine.geomechanics(df_clean, proppant_eff, srv_scale)
curves = engine.forecast(geo["conductivity_index"], noise, tensile_pct, rate_limit)
 
# --------------------------------------------------------------------------
# Header and status banners
# --------------------------------------------------------------------------
st.title("GeoFlux Reservoir Suite")
st.subheader(f"Prototype analytics core ({env})")
st.warning("ILLUSTRATIVE PROTOTYPE: outputs are not validated and must not be used "
           "for safety, reserves, regulatory or investment decisions.")
if is_synthetic:
    st.warning(f"DATA SOURCE: {source_label}")
else:
    st.info(f"DATA SOURCE: {source_label}")
for w in load_warnings:
    st.caption(f"Data note: {w}")
st.markdown("---")
 
b = geo["b"]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("SRV (hull x scale)", f"{geo['srv']:,.0f} m³")
m2.metric("Conductivity index (relative)", f"{geo['conductivity_index']:.2f}")
m3.metric("b-value (Aki ML)",
          f"{b['b']:.2f} ± {b['sigma']:.2f}" if b else "n/a (too few events)")
m4.metric("Time above rate limit", f"{curves['life_years']} yr")
m5.metric("Illustrative 10-yr cumulative", f"{curves['cum_boe']:,.0f} BOE")
if not geo["hull_ok"]:
    st.caption("SRV is 0: fewer than 4 events or degenerate geometry (hull could not be computed).")
if curves["horizon_capped"]:
    st.caption("The rate never fell below the limit within the 10-year horizon; "
               "cumulative volume is capped at 10 years.")
elif curves["life_years"] == 0:
    st.caption("The rate is below the economic limit from day 1: cumulative volume is zero.")
if b:
    msg = f"b-value estimated from {b['n_used']} events above Mc = {b['mc']:.2f}."
    if is_synthetic:
        msg += f" Synthetic truth: b = {b_true:.1f} (a quick sanity check of the estimator)."
    st.caption(msg)
st.markdown("---")
 
# --------------------------------------------------------------------------
# Channel A: forecast (placeholder)
# --------------------------------------------------------------------------
st.markdown("### Channel A: Illustrative decline forecast (placeholder model)")
st.caption("Not calibrated to any reservoir. No porosity, permeability, pressure or fluid "
           "properties are used. Treat the shape as a demonstration only.")
idx = np.round(curves["timeline"], 2)
c1, c2 = st.columns(2)
c1.area_chart(pd.DataFrame({"Hydrocarbon rate (BOE/d)": curves["rate"]}, index=idx))
c2.area_chart(pd.DataFrame({"Produced water (BWPD)": curves["water"]}, index=idx))
st.markdown("---")
 
# --------------------------------------------------------------------------
# Channel B: seismology scrutiny view
# --------------------------------------------------------------------------
st.markdown("### Channel B: Seismicity and filter validation")
v1, v2, v3 = st.columns(3)
 
with v1:
    st.caption("Frequency-magnitude: log10 of cumulative count N(>=M)")
    m_raw, m_clean = energy_to_magnitude(raw_df["Energy_J"].to_numpy()), geo["magnitudes"]
    grid = np.linspace(min(m_raw.min(), m_clean.min()), max(m_raw.max(), m_clean.max()), 25)
 
    def cum_log(m):
        n = np.array([(m >= g).sum() for g in grid], dtype=float)
        return np.where(n > 0, np.log10(np.maximum(n, 1.0)), np.nan)
 
    st.line_chart(pd.DataFrame({"Unfiltered": cum_log(m_raw), "Filtered": cum_log(m_clean)},
                               index=np.round(grid, 2)))
 
with v2:
    st.caption("Stress ESTIMATE from empirical gradients (MPa). Not an inversion.")
    st.bar_chart(pd.DataFrame({"MPa": [stress["Sv"], stress["Shmax"], stress["Shmin"]]},
                              index=["Sv", "SHmax", "Shmin"]))
    st.caption(f"Computed at median event depth {stress['depth_m']:,.0f} m.")
 
with v3:
    st.caption("Illustrative proppant decay profile (placeholder, not a model)")
    d_axis = np.linspace(0, 300, 100)
    st.area_chart(pd.DataFrame(
        {"Proppant (lb/ft²)": (proppant_eff * 0.1) * np.exp(-d_axis / (100 + geo["conductivity_index"]))},
        index=np.round(d_axis, 1)))
st.markdown("---")
 
# --------------------------------------------------------------------------
# Filter audit
# --------------------------------------------------------------------------
st.markdown("### Filter audit")
a1, a2 = st.columns(2)
with a1:
    counts = pd.DataFrame({
        "Raw": [int((raw_df["Mechanism"] == k).sum()) for k in VALID_MECHANISMS],
        "After filters": [int((df_clean["Mechanism"] == k).sum()) for k in VALID_MECHANISMS],
    }, index=VALID_MECHANISMS)
    st.bar_chart(counts)
with a2:
    st.write(f"Distance buffer: removed **{filter_info['removed_by_buffer']}** events within "
             f"{spatial_buffer:g} m of Easting = {workings_easting:g} m (planar approximation).")
    st.write(f"Label filter ({'ON' if drop_closure else 'OFF'}): removed "
             f"**{filter_info['removed_by_label']}** events labelled 'Closure'.")
    st.write(f"Closure share of raw catalogue energy: **{stress['closure_energy_fraction']:.1%}**")
    st.caption("The label filter trusts the 'Mechanism' column. It performs no waveform "
               "analysis, and 'Closure' is not the same thing as blasting or machinery noise.")
st.markdown("---")
 
with st.expander("Known limitations of this prototype"):
    st.markdown(
        "- No reservoir properties are modelled; the forecast is a placeholder curve.\n"
        "- SRV is a convex-hull volume (sensitive to outliers; no uncertainty).\n"
        "- The conductivity index is a relative proxy, not Fcd.\n"
        "- Stress values are gradient-based estimates, not a focal-mechanism inversion.\n"
        "- The mine buffer is a planar Easting offset, not 3-D distance to workings.\n"
        "- No hazard metrics (Mmax, rockburst, ground motion) are computed.\n"
        "- No economics: no prices, costs, royalties, tax or discounting.")
 
st.subheader("Event catalogue (first 20 rows)")
st.dataframe(raw_df.head(20))
 
# --------------------------------------------------------------------------
# CHANGELOG (Phase 0)
# - Removed the unconditional "PRO-FORMA REVIEW PASSED" banner.
# - Data source is always labelled; a failed upload now shows an error instead of
#   silently switching to synthetic data. Required columns are validated.
# - Fixed Ultra-Deep substring bug: regimes are looked up by exact key. Ultra-Deep
#   now actually uses its own depth (5,000 m) and water exponent (0.50).
# - b-value: Aki ML estimator with Mc (max-curvature + 0.2) and 0.434 factor,
#   reported with +/- sigma. Removed arbitrary clip and Ultra-Deep x0.85 factor.
# - Magnitude relation corrected to M = (log10 E - 4.8) / 1.5 (E in joules).
# - Removed the fake "Error Bound" series from the frequency-magnitude plot.
# - Synthetic magnitudes now follow Gutenberg-Richter with a known b.
# - Synthetic depth now follows the selected envelope.
# - Distance buffer: working Easting is a user input (was hardcoded 500).
# - Renamed "waveform discriminator" to what it is: a label filter.
# - SRV: arbitrary x10.5/14.2/18.5 multipliers replaced by a user scaling factor (1.0).
# - "Fcd (mD-ft)" renamed "Conductivity index (relative)".
# - Forecast: SRV no longer steepens decline; Arps b capped at 2; cutoff logic fixed
#   (a well never reaching the limit is capped at 10 yr; one below it on day 1 gets 0).
# - Stress ellipse plot replaced with an honest Sv/SHmax/Shmin bar chart.
# - Machinery-noise chart now compares raw vs filtered counts per mechanism.
# - Removed global np.random.seed; uses a local default_rng.
# --------------------------------------------------------------------------
 
