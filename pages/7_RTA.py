# geoflux/pages/7_RTA.py - GeoFlux Rate-Transient Analysis, lite (PROTOTYPE)
#
# Not validated. Single-phase liquid only. Results only reflect the data and inputs you supply.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import streamlit as st

import geoflux_agg as agg
import geoflux_rta as rta
import geoflux_rtadata as rd
import geoflux_runs as runs
import tabs_rta

try:
    st.set_page_config(page_title="GeoFlux: RTA (prototype)", layout="wide")
except Exception:
    pass


@st.cache_data(show_spinner=False)
def demo(kind):
    return rd.make_synthetic_rta(noise=0.03, bdf_at_days=500 if kind != "Linear flow only" else None)


st.sidebar.header("Data")
f = st.sidebar.file_uploader("Rate and flowing pressure (.csv)", type=["csv", "txt"],
                             help="Date or Time_days, Rate_sm3d (or Rate_bbld), Pwf_MPa (or Pwf_psi). Period averages.")
kind = st.sidebar.radio("Demo (used when nothing is uploaded)", ["Linear flow only", "Linear then boundary-dominated"])
notes, truth = [], None
if f is not None:
    hist, err, notes = rd.load_rta_history(f)
    if err:
        st.error(f"Upload failed: {err}")
        st.stop()
    is_demo = False
else:
    (hist, truth), is_demo = demo(kind), True

st.sidebar.header("Fluid and rock")
pi = st.sidebar.number_input("Initial reservoir pressure (MPa)", 1.0, 200.0, 40.0, 1.0)
bo = st.sidebar.number_input("Oil FVF Bo (rm3/sm3)", 1.0, 3.0, 1.2, 0.01)
phi = st.sidebar.number_input("Porosity", 0.01, 0.4, 0.08, 0.01)
ct = st.sidebar.number_input("Total compressibility (1/MPa)", 1e-5, 0.02, 2e-3, 1e-4, format="%.5f")
mu = st.sidebar.number_input("Viscosity (cP)", 0.05, 100.0, 1.0, 0.1)
sw = st.sidebar.number_input("Water saturation", 0.0, 0.9, 0.3, 0.05)
st.sidebar.header("Uncertainty (%)")
rel_phi = st.sidebar.slider("Porosity", 0, 40, 10) / 100.0
rel_ct = st.sidebar.slider("Compressibility", 0, 40, 15) / 100.0
rel_mu = st.sidebar.slider("Viscosity", 0, 40, 5) / 100.0
rel_rate = st.sidebar.slider("Rate measurement", 0, 20, 3) / 100.0
rel_pwf = st.sidebar.slider("Flowing pressure measurement", 0, 10, 2) / 100.0
n_mc = st.sidebar.slider("Monte Carlo samples", 100, 1000, 300, 100)
st.sidebar.header("Completion (optional)")
h_m = st.sidebar.number_input("Net pay thickness (m)", 1.0, 300.0, 30.0, 1.0)
n_frac = st.sidebar.number_input("Number of fractures", 1, 500, 40, 1)
known = {"None": None, "Permeability known": "k", "Fracture half-length known": "xf"}[
    st.sidebar.radio("Known quantity", ["None", "Permeability known", "Fracture half-length known"])]
known_value = None
if known == "k":
    known_value = st.sidebar.number_input("Permeability (md)", 1e-6, 100.0, 0.007, 0.001, format="%.6f")
elif known == "xf":
    known_value = st.sidebar.number_input("Fracture half-length (m)", 1.0, 500.0, 60.0, 5.0)

d = rta.prepare(hist, pi, bo)
n = len(d["tmb"])
if n < 8:
    st.error("Fewer than 8 usable points (rate above zero and flowing pressure below the initial pressure).")
    st.stop()
slope = rta.loglog_slope(d["tmb"], d["rnp"], 5)
auto_win = rta.auto_linear_window(slope)
jw = (2, n - 1)
joint = rta.joint_fit(d["x_sp"], d["tmb"], d["rnp"], *jw)
boundary = rta.boundary_detected(joint)
st.sidebar.header("Linear-flow window")
manual = st.sidebar.radio("Window", ["Automatic", "Manual"]) == "Manual"
if manual or auto_win is None:
    a0, a1 = auto_win if auto_win else (3, min(n, 20) - 1)
    p0 = st.sidebar.slider("First point", 1, n - 4, a0 + 1)
    p1 = st.sidebar.slider("Last point", p0 + 4, n, max(p0 + 4, a1 + 1))
    win, was_auto = (p0 - 1, p1 - 1), False
    if auto_win is None and not manual:
        st.sidebar.warning("No clear linear-flow stretch was found automatically. Pick a window by hand.")
else:
    win, was_auto = auto_win, True

lin = rta.linear_fit(d["x_sp"], d["rnp"], *win)
fallback = auto_win is None and not manual
if boundary:
    if joint["a"] <= 0:
        st.error("The joint fit has a non-positive linear slope, so there is no linear-flow solution.")
        st.stop()
    cs = rta.a_sqrt_k_mc(hist, pi, bo, jw, phi, ct, mu, rel_rate, rel_pwf, rel_phi, rel_ct, rel_mu, n_mc, 0, joint=True)
else:
    if lin["m"] <= 0:
        st.error("The straight line has a non-positive slope, so there is no linear-flow solution. Pick another window.")
        st.stop()
    cs = rta.a_sqrt_k_mc(hist, pi, bo, win, phi, ct, mu, rel_rate, rel_pwf, rel_phi, rel_ct, rel_mu, n_mc, 0)
if len(cs) < 20:
    st.error("The fit is unstable under measurement error. Choose a cleaner window.")
    st.stop()

inputs = {"history": hist, "pi": pi, "bo": bo, "phi": phi, "ct": ct, "mu": mu, "sw": sw, "window": list(win),
          "method": "joint" if boundary else "linear",
          "uncertainty": {"phi": rel_phi, "ct": rel_ct, "mu": rel_mu, "rate": rel_rate, "pwf": rel_pwf},
          "h_m": h_m, "n_frac": int(n_frac), "known": known, "known_value": known_value, "n_mc": n_mc}
rec = runs.make_record("rta_lite", inputs, {"a_sqrt_k": agg.percentiles(cs), "slope_m": lin["m"], "r2": lin["r2"]}, 0)

st.title("GeoFlux: Rate-Transient Analysis (lite)")
st.warning("PROTOTYPE: not validated. Single-phase liquid only: no gas, no multiphase flow. Results only reflect the "
           "data and inputs you supply.")
(st.warning if is_demo else st.info)("DATA SOURCE: " + (
    f"SYNTHETIC DEMONSTRATION WELL ({kind}; invented, not real). The demo's true properties match the default inputs."
    if is_demo else f"UPLOADED: {f.name} ({len(hist)} records)"))
for n_ in notes:
    st.caption(f"Data note: {n_}")

S = SimpleNamespace(d=d, slope=slope, win=win, auto=was_auto, lin=lin, cs=cs, joint=joint, boundary=boundary,
                    fallback=fallback, jw=jw, phi=phi, ct=ct, mu=mu, bo=bo, sw=sw,
                    h_m=h_m, n_frac=int(n_frac), known=known, known_value=known_value, truth=truth, rec=rec)
tabs = st.tabs(["Diagnostics", "Linear flow", "Boundary-dominated", "Notes & limits", "Run record"])
with tabs[0]:
    tabs_rta.render_diagnostics(S)
with tabs[1]:
    tabs_rta.render_linear(S)
with tabs[2]:
    tabs_rta.render_bdf(S)
with tabs[3]:
    tabs_rta.render_notes(S)
with tabs[4]:
    tabs_rta.render_record(S)

# END OF FILE: 7_RTA.py