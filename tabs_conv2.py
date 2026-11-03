"""GeoFlux conventional engine: gas P/Z and oil material balance tabs."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import geoflux_agg as agg
import geoflux_mb as mb
import geoflux_mbdata as md
import geoflux_pvt as pv
import tabs_conv

BCF = 35.3147 / 1e9                              # sm3 -> Bcf
MMSTB = 6.28981 / 1e6                            # sm3 -> MMSTB


def render_pz():
    src = st.radio("Data", ["Demo: straight line", "Demo: curved", "Upload"], horizontal=True, key="pz_src")
    c = st.columns(3)
    t_c = c[0].number_input("Reservoir temperature (degC)", 20.0, 250.0, 100.0, 5.0, key="pz_t")
    gam = c[1].number_input("Gas gravity (air = 1)", 0.55, 1.2, 0.70, 0.01, key="pz_g")
    p_ab = c[2].number_input("Abandonment pressure (MPa)", 0.5, 20.0, 5.0, 0.5, key="pz_pab")
    truth = None
    if src == "Upload":
        f = st.file_uploader("History: Date or Time_days, Pressure_MPa, Gp_sm3", type=["csv", "txt"], key="pz_file")
        if f is None:
            st.info("Upload a pressure and cumulative-gas history, or pick a demo.")
            return
        h, err, notes = md.load_mb_history(f)
        if err:
            st.error(err)
            return
        for n_ in notes:
            st.caption(f"Data note: {n_}")
    else:
        h, truth = md.make_synthetic_gas_mb(curvature=0.0 if "straight" in src else 0.3, noise=0.01)
        st.warning("SYNTHETIC DEMONSTRATION RESERVOIR (invented, not real). True G = 10 billion sm3.")
    z = pv.z_papay(h["Pressure_MPa"], t_c, gam)
    try:
        r = mb.pz_analysis(h["Gp_sm3"], h["Pressure_MPa"], z, p_ab, float(pv.z_papay(p_ab, t_c, gam)), seed=0)
    except ValueError as exc:
        st.error(str(exc))
        return
    if not np.isfinite(r["G"]):
        st.error("The p/z trend does not fall with production, so no gas-in-place intercept exists. Check the data.")
        return
    g, rec = agg.percentiles(r["G_samples"] * BCF), agg.percentiles(r["recoverable_samples"] * BCF)
    k = st.columns(5)
    for col, lab, v in zip(k, ["G P90 (low)", "G P50", "G P10 (high)", "Recoverable P50"], [g["P90"], g["P50"], g["P10"], rec["P50"]]):
        col.metric(lab, f"{v:,.1f} Bcf")
    k[4].metric("Recovery factor P50", f"{float(np.median(r['recoverable_samples'] / r['G_samples'])) * 100:.0f} %")
    gp_bcf = h["Gp_sm3"].to_numpy() * BCF
    pz = (h["Pressure_MPa"] / z).to_numpy()
    xs = np.linspace(0, r["G"] * BCF, 50)
    line = pd.DataFrame({"Gp": xs, "pz": r["a"] + r["b"] / BCF * xs})
    pts = alt.Chart(pd.DataFrame({"Gp": gp_bcf, "pz": pz})).mark_circle(size=70, color="#d62728").encode(
        x=alt.X("Gp:Q", title="Cumulative gas Gp (Bcf)"), y=alt.Y("pz:Q", title="p/z (MPa)"))
    st.altair_chart((pts + alt.Chart(line).mark_line().encode(x="Gp:Q", y="pz:Q")).properties(
        width="container", height=300, title="P/Z plot with the straight-line fit extended to the axis"))
    st.caption(f"Fit R2 = {r['r2']:.4f} on {r['n']} points. G from all points {r['G'] * BCF:,.1f} Bcf; "
               f"early half {r['G_early'] * BCF:,.1f}; late half {r['G_late'] * BCF:,.1f}.")
    if r["curved"]:
        st.warning(f"The late slope is {r['slope_ratio']:.2f} times the early slope, so the data is not a straight line "
                   "(possible water influx, compaction, or data error). The G range above assumes a straight line and is "
                   "overconfident here. Compare the early-half and late-half values as a bracket, and investigate the cause.")
    if truth is not None:
        st.caption("Demo truth: G = 10.0 Bsm3 = 353 Bcf" + (". The curved demo's straight-line G is deliberately wrong." if r["curved"] else "."))
    st.warning("Assumes volumetric depletion (no water influx), measured average pressures, and an approximate z-factor "
               "(Papay correlation: use laboratory z-factors if you have them). A resource estimate, not reserves.")
    tabs_conv.record_box("pz_material_balance", {"history": h, "t_c": t_c, "gas_gravity": gam, "p_abandon": p_ab},
                         {"G_bcf": g, "recoverable_bcf": rec, "r2": r["r2"], "curved": r["curved"]}, 0, "pz")


def _oil_source():
    src = st.radio("Data", ["Demo: depletion", "Demo: water influx", "Demo: gas cap", "Upload"], horizontal=True, key="mb_src")
    if src != "Upload":
        kw = {"Demo: depletion": {}, "Demo: water influx": {"C": 2.0e3}, "Demo: gas cap": {"m": 0.3}}[src]
        h, pvt, truth = md.make_synthetic_oil_mb(noise=0.005, **kw)
        st.warning("SYNTHETIC DEMONSTRATION RESERVOIR (invented, not real). True N = 50 million sm3 (314 MMSTB)"
                   + (f", aquifer constant C = {truth['C']:,.0f}." if truth["C"] else f", gas-cap ratio m = {truth['m']}." if truth["m"] else "."))
        return h, pvt, truth
    f1 = st.file_uploader("History: Date or Time_days, Pressure_MPa, Np_sm3, Gp_sm3, Wp_sm3", type=["csv", "txt"], key="mb_hist")
    f2 = st.file_uploader("PVT table: Pressure_MPa, Bo, Rs, Bg", type=["csv", "txt"], key="mb_pvt")
    if f1 is None or f2 is None:
        st.info("Upload both files, or pick a demo.")
        return None
    h, err, notes = md.load_mb_history(f1)
    pvt, err2 = md.load_pvt(f2)
    if err or err2:
        st.error(err or err2)
        return None
    for n_ in notes:
        st.caption(f"Data note: {n_}")
    return h, pvt, None


def render_oil_mb():
    got = _oil_source()
    if got is None:
        return
    h, pvt, truth = got
    c = st.columns(5)
    pi = c[0].number_input("Initial pressure (MPa)", 1.0, 100.0, float(truth["pi"]) if truth else float(h["Pressure_MPa"].max()), 0.5, key="mb_pi")
    swi = c[1].number_input("Connate water Swi", 0.0, 0.6, 0.2, 0.01, key="mb_swi")
    cw = c[2].number_input("Water compressibility (1/MPa)", 0.0, 0.005, 4.4e-4, 1e-5, format="%.6f", key="mb_cw")
    cf = c[3].number_input("Pore compressibility (1/MPa)", 0.0, 0.005, 5.8e-4, 1e-5, format="%.6f", key="mb_cf")
    method = c[4].selectbox("Model", ["depletion", "aquifer", "gascap"], key="mb_method",
                            format_func={"depletion": "Depletion (no gas cap, no aquifer)", "aquifer": "Steady-state aquifer",
                                         "gascap": "Unknown gas cap"}.get)
    e = st.columns(3)
    p_rel = e[0].slider("Pressure measurement error (%)", 0.1, 3.0, 0.5, 0.1, key="mb_prel") / 100.0
    v_rel = e[1].slider("Cumulative volume error (%)", 0.0, 5.0, 1.0, 0.5, key="mb_vrel") / 100.0
    n_mc = e[2].slider("Monte Carlo fits", 100, 1000, 300, 100, key="mb_n")
    props = {"swi": swi, "cw": cw, "cf": cf, "m": 0.0}
    try:
        d = mb.ho_prepare(h, pvt, pi, **props)
        if len(d["F"]) < 4:
            st.error("Fewer than 4 usable points below the initial pressure.")
            return
        Et = d["eo"] + d["efw"]
        if method == "depletion":
            r = mb.ho_depletion(d["F"], Et)
            fitF = r["N"] * Et
        elif method == "aquifer":
            r = mb.ho_aquifer(d["F"], Et, d["intdp"])
            fitF = r["N"] * Et + r["C"] * d["intdp"]
        else:
            r = mb.ho_gascap(d["F"], Et, d["eg"])
            fitF = r["N"] * (Et + r["m"] * d["eg"])
        mc = mb.ho_monte_carlo(h, pvt, pi, method, p_rel, v_rel, n_mc, 0, **props)
    except (ValueError, np.linalg.LinAlgError) as exc:
        st.error(f"Could not fit: {exc}")
        return
    if len(mc["N_samples"]) < 20:
        st.error("The fit is unstable under measurement error. Check pressures, PVT and the chosen model.")
        return
    n_p = agg.percentiles(mc["N_samples"] * MMSTB)
    k = st.columns(5)
    k[0].metric("N best fit", f"{r['N'] * MMSTB:,.0f} MMSTB")
    k[1].metric("N P90 (low)", f"{n_p['P90']:,.0f} MMSTB")
    k[2].metric("N P50", f"{n_p['P50']:,.0f} MMSTB")
    k[3].metric("N P10 (high)", f"{n_p['P10']:,.0f} MMSTB")
    if method == "aquifer":
        k[4].metric("Aquifer constant C", f"{r['C']:,.0f} rm3/(MPa day)")
    elif method == "gascap":
        k[4].metric("Gas-cap ratio m", f"{r['m']:.2f}")
    else:
        k[4].metric("Fit R2", f"{r['r2']:.4f}")
    df = pd.DataFrame({"Np_MMSTB": d["np"] * MMSTB, "Measured": d["F"] / 1e6, "Fitted": fitF / 1e6}).melt("Np_MMSTB", var_name="Series", value_name="F")
    st.altair_chart(alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Np_MMSTB:Q", title="Cumulative oil Np (MMSTB)"), y=alt.Y("F:Q", title="Underground withdrawal F (million rm3)"),
        color="Series:N").properties(width="container", height=280, title="Havlena-Odeh: measured and fitted withdrawal"))
    st.write(mb.drive_hint(d["F"], Et))
    if method == "aquifer":
        st.warning("N and the aquifer constant are strongly correlated, so the N range is wide. That is the physics, not a "
                   "bug: more pressure history, or independent aquifer information, is the only cure.")
    if method == "depletion":
        st.caption("If a water drive exists, the depletion fit can overstate N several-fold. Try the aquifer model and "
                   "compare.")
    if method == "gascap":
        st.caption("The gas-cap fit's range runs a little narrow in testing, and it ignores the gas cap's small effect on "
                   "the rock and water expansion term.")
    st.warning("The range reflects only the measurement errors you set (pressure, volumes), not PVT error, model choice or "
               "pressure-averaging error. A resource estimate, not reserves.")
    tabs_conv.record_box("havlena_odeh", {"history": h, "pvt": pvt, "pi": pi, "model": method, "p_rel": p_rel, "vol_rel": v_rel, **props},
                         {"N_mmstb": n_p, "best_fit_mmstb": r["N"] * MMSTB}, 0, "mb")


# END OF FILE: tabs_conv2.py