"""GeoFlux RTA-lite: tab renderers."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import geoflux_agg as agg
import geoflux_rta as rta
import geoflux_runs as runs

MMSTB = 6.28981 / 1e6


def render_diagnostics(S):
    d, slope = S.d, S.slope
    deriv = slope * d["rnp"]
    long = pd.concat([pd.DataFrame({"tmb": d["tmb"], "value": d["rnp"], "Series": "RNP"}),
                      pd.DataFrame({"tmb": d["tmb"], "value": deriv, "Series": "Derivative d(RNP)/d(ln tmb)"})]).dropna()
    st.altair_chart(alt.Chart(long).mark_point(size=60, filled=True).encode(
        x=alt.X("tmb:Q", scale=alt.Scale(type="log"), title="Material balance time (days)"),
        y=alt.Y("value:Q", scale=alt.Scale(type="log"), title="MPa d / m3"), color="Series:N").properties(
        width="container", height=300, title="Rate-normalised pressure and its derivative"))
    tr = pd.DataFrame({"tmb": d["tmb"], "Slope": slope}).dropna()
    rules = alt.Chart(pd.DataFrame({"y": [0.5, 1.0], "label": ["linear flow (0.5)", "boundary-dominated (1.0)"]})).mark_rule(
        strokeDash=[5, 4], color="gray").encode(y="y:Q")
    st.altair_chart((alt.Chart(tr).mark_line(point=True).encode(
        x=alt.X("tmb:Q", scale=alt.Scale(type="log"), title="Material balance time (days)"),
        y=alt.Y("Slope:Q", title="Log-log slope of RNP")) + rules).properties(
        width="container", height=220, title="Flow regime trace (about 0.5 linear, about 1 boundary-dominated)"))
    counts = pd.Series(rta.classify(slope)).value_counts().rename("Points").to_frame()
    st.dataframe(counts)
    st.caption("The slope trace is noisy by nature: read the trend, not single points. Confirming a flow regime properly "
               "also needs the pressure transient and a geological check.")


def render_linear(S):
    d, lin, i0, i1, jt = S.d, S.lin, S.win[0], S.win[1], S.joint
    c_lin = rta.a_sqrt_k(lin["m"], S.phi, S.ct, S.mu) if lin["m"] > 0 else float("nan")
    c_joint = rta.a_sqrt_k(jt["a"], S.phi, S.ct, S.mu) if jt["a"] > 0 else float("nan")
    if S.boundary:
        st.warning(f"Boundary effects detected (boundary term t = {jt['t_c']:.0f}). A straight line through the early data "
                   f"would give A sqrt(k) of about {c_lin:,.0f}; the joint linear + boundary fit gives {c_joint:,.0f}. "
                   "The result below uses the joint fit, because a linear-only window is contaminated.")
    elif S.fallback:
        st.warning("No clean linear-flow stretch was found automatically, so a default window is shown. Check the "
                   "diagnostics and choose the window by hand; the result may be contaminated by other flow regimes.")
    p = agg.percentiles(S.cs)
    k = st.columns(4)
    k[0].metric("A sqrt(k) P90 (low)", f"{p['P90']:,.0f}")
    k[1].metric("A sqrt(k) P50", f"{p['P50']:,.0f}")
    k[2].metric("A sqrt(k) P10 (high)", f"{p['P10']:,.0f}")
    k[3].metric("Method", "Joint linear + boundary" if S.boundary else f"Linear window (R2 {lin['r2']:.4f})")
    xs = np.linspace(float(np.min(d["x_sp"])), float(np.max(d["x_sp"])), 50)
    idx = np.arange(len(d["rnp"]))
    if S.boundary:
        st.caption(f"Joint fit over points {S.jw[0] + 1} to {S.jw[1] + 1}: RNP = {jt['b']:.4f} + {jt['a']:.4f} x + {jt['c']:.3e} tmb "
                   "(x = square-root superposition time). Units of A sqrt(k): m2 sqrt(md).")
        fit_pts = pd.DataFrame({"x": d["x_sp"], "RNP": jt["b"] + jt["a"] * d["x_sp"] + jt["c"] * d["tmb"]})
        df = pd.DataFrame({"x": d["x_sp"], "RNP": d["rnp"], "Used": idx >= S.jw[0]})
        line = fit_pts
    else:
        st.caption(f"Window: points {i0 + 1} to {i1 + 1} ({'automatic' if S.auto else 'manual'}), {lin['n']} points. "
                   f"Slope {lin['m']:.4f} MPa d / m3 per sqrt(d). Units of A sqrt(k): m2 sqrt(md).")
        df = pd.DataFrame({"x": d["x_sp"], "RNP": d["rnp"], "Used": (idx >= i0) & (idx <= i1)})
        line = pd.DataFrame({"x": xs, "RNP": lin["m"] * xs + lin["b"]})
    st.altair_chart((alt.Chart(df).mark_circle(size=70).encode(
        x=alt.X("x:Q", title="Square-root superposition time (sqrt(d))"), y=alt.Y("RNP:Q", title="RNP (MPa d / m3)"),
        color="Used:N") + alt.Chart(line).mark_line(color="black").encode(x="x:Q", y="RNP:Q")).properties(
        width="container", height=300, title="Joint fit" if S.boundary else "Linear-flow straight line"))
    if not S.boundary:
        with st.expander("Why not material balance time?"):
            f2 = rta.linear_fit(np.sqrt(d["tmb"]), d["rnp"], i0, i1)
            c2 = rta.a_sqrt_k(f2["m"], S.phi, S.ct, S.mu)
            st.write(f"Using sqrt(material balance time) instead would give A sqrt(k) of about {c2:,.0f}, versus "
                     f"{p['P50']:,.0f} here. In testing that shortcut ran roughly 13% low when pressure varies, because it "
                     "only approximates a constant-rate response. Superposition time is exact for linear flow.")
    st.markdown("**Only the product A sqrt(k) is identifiable.** Fracture area and permeability trade off:")
    h, n = S.h_m, S.n_frac
    if S.known == "k":
        xf = S.cs / np.sqrt(S.known_value) / (2.0 * h * n)
        q = agg.percentiles(xf)
        st.metric("Fracture half-length xf (given k)", f"{q['P50']:,.0f} m", f"P90 {q['P90']:,.0f} to P10 {q['P10']:,.0f}")
    elif S.known == "xf":
        kk = (S.cs / (2.0 * S.known_value * h * n)) ** 2
        q = agg.percentiles(kk)
        st.metric("Permeability k (given xf)", f"{q['P50']:.5f} md", f"P90 {q['P90']:.5f} to P10 {q['P10']:.5f}")
    tt = pd.DataFrame(rta.xf_k_tradeoff(p["P50"], h, n, [20, 40, 60, 80, 100, 150, 200]))
    st.altair_chart(alt.Chart(tt).mark_line(point=True).encode(
        x=alt.X("xf_m:Q", title="Fracture half-length xf (m)"), y=alt.Y("k_md:Q", scale=alt.Scale(type="log"), title="Permeability (md)")).properties(
        width="container", height=220, title=f"Every point fits the data equally well (h = {h:g} m, {n} fractures)"))
    st.dataframe(tt.round(6), hide_index=True)
    if S.truth:
        st.caption(f"Demo truth: A sqrt(k) = {S.truth['c']:,.0f}.")
    st.warning("The range includes measurement error and the porosity, compressibility and viscosity uncertainty you set; "
               "property uncertainty usually dominates. It excludes model error (are you really in linear flow?), "
               "multiphase flow, and changing fracture conductivity. In testing the joint-fit range ran a little narrow "
               "(70 to 75% coverage for an 80% range). Liquid, single-phase, slightly compressible only.")


def render_bdf(S):
    d, n = S.d, len(S.d["tmb"])
    st.write("Boundary-dominated flow " + (f"**is detected** by the joint fit over all the data (t = {S.joint['t_c']:.0f})."
                                           if S.boundary else "is **not detected** by the joint fit over all the data."))
    c = st.columns(2)
    j0 = c[0].slider("First point of the late-time window", 1, n - 6, max(1, int(n * 0.55)), key="rta_b0")
    j1 = c[1].slider("Last point", j0 + 6, n, n, key="rta_b1")
    i0, i1 = j0 - 1, j1 - 1
    mean_slope = float(np.nanmean(S.slope[i0:i1 + 1])) if np.isfinite(S.slope[i0:i1 + 1]).any() else float("nan")
    try:
        f = rta.bdf_fit(d["tmb"], d["rnp"], i0, i1, S.ct, S.sw, S.bo, x_sp=d["x_sp"])
    except ValueError as exc:
        st.info(str(exc))
        return
    k = st.columns(4)
    k[0].metric("Pore volume Vp", f"{f['pore_volume_m3'] / 1e6:,.3f} million m3")
    k[1].metric("Hydrocarbon pore volume", f"{f['hcpv_m3'] / 1e6:,.3f} million m3")
    k[2].metric("Oil in contact", f"{f['in_place_sm3'] * MMSTB:,.2f} MMSTB")
    k[3].metric("Slope uncertainty", f"{f['vp_rel_se'] * 100:.0f} %")
    if not np.isfinite(mean_slope) or mean_slope < 0.8:
        st.warning(f"The average log-log slope in this window is {mean_slope:.2f}, below about 0.8, so boundary-dominated "
                   "flow is NOT clearly visible. Treat the volume as unreliable; it is not a drainage volume.")
    st.altair_chart(alt.Chart(pd.DataFrame({"tmb": d["tmb"], "RNP": d["rnp"]})).mark_circle(size=70, color="#d62728").encode(
        x=alt.X("tmb:Q", title="Material balance time (days)"), y=alt.Y("RNP:Q", title="RNP (MPa d / m3)")).properties(
        width="container", height=280, title="RNP against material balance time"))
    st.caption(f"Fit over points {j0} to {j1}: RNP = {f['b']:.4f} + {f['a']:.4f} x + {f['c']:.3e} tmb, where x is superposition time.")
    if S.truth and S.truth.get("vp_m3"):
        st.caption(f"Demo truth: pore volume = {S.truth['vp_m3'] / 1e6:,.3f} million m3.")
    st.warning("Only meaningful if boundary-dominated flow has actually been reached. The estimate is the volume the well "
               "has felt so far. It assumes the same fluid properties throughout and a closed drainage volume.")


def render_notes(S=None):
    st.markdown(
        "**What this does**\n"
        "- Rate-normalised pressure and material balance time diagnostics, with a log-log flow-regime trace.\n"
        "- Linear-flow analysis on square-root superposition time, giving A sqrt(k) with an honest range.\n"
        "- An xf-k trade-off, because only A sqrt(k) is identifiable. Give k or xf and the other follows.\n"
        "- A boundary-dominated fit for pore volume, when that regime is visible.\n\n"
        "**What it does not do**\n"
        "- No gas (needs pseudo-pressure and pseudo-time), no multiphase flow, no type-curve matching, no "
        "numerical models, no changing fracture conductivity or interference.\n"
        "- Needs flowing-pressure data. Without it, use the decline-curve pages instead.\n\n"
        "**How it was tested**\n"
        "- The linear-flow formula agrees with an independent numerical solution of the diffusion equation to 0.01%.\n"
        "- On synthetic wells it recovers the true A sqrt(k) within about 1%, and the ranges cover the truth 83 to 87% of "
        "the time. Real wells will be messier: those figures are best-case.")


def render_record(S):
    st.write(f"Engine version **{S.rec['engine_version']}**, seed **{S.rec['seed']}**.")
    st.code(S.rec["inputs_sha256"], language=None)
    st.download_button("Download run record (JSON)", runs.record_json(S.rec), "geoflux_rta_record.json", "application/json")


# END OF FILE: tabs_rta.py