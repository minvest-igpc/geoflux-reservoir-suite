"""GeoFlux conventional engine: volumetrics tab and the run-record helper."""
import altair as alt
import pandas as pd
import streamlit as st

import geoflux_agg as agg
import geoflux_runs as runs
import geoflux_vol as vol

LABELS = {"area_km2": "Area (km2)", "h_m": "Net pay thickness (m)", "phi": "Porosity (fraction)",
          "sw": "Water saturation (fraction)", "rf": "Recovery factor (fraction)"}


def record_box(method, inputs, outputs, seed, name):
    rec = runs.make_record(method, inputs, outputs, seed)
    with st.expander("Run record"):
        st.code(rec["inputs_sha256"], language=None)
        st.download_button("Download run record (JSON)", runs.record_json(rec), f"geoflux_{name}_record.json",
                           "application/json", key=f"dl_{name}")


def render_volumetrics():
    fluid = st.radio("Fluid", ["Oil", "Gas"], horizontal=True, key="vol_fluid")
    oil = fluid == "Oil"
    base = vol.OIL_SPEC if oil else vol.GAS_SPEC
    unit, conv, per = ("MMSTB", vol.SM3_TO_STB / 1e6, "oil") if oil else ("Bcf", vol.SM3_TO_SCF / 1e9, "gas")
    labels = {**LABELS, "fvf": "Oil FVF Bo (rm3/sm3)" if oil else "Gas FVF Bg (rm3/sm3)"}
    st.markdown("**Inputs: low, most likely, high** (triangular distributions)")
    head = st.columns([2, 1, 1, 1])
    for c, t in zip(head, ["Input", "Low", "Most likely", "High"]):
        c.caption(t)
    spec = {}
    for k, (lo, mode, hi) in base.items():
        c = st.columns([2, 1, 1, 1])
        c[0].write(labels[k])
        spec[k] = tuple(c[i + 1].number_input(labels[k] + f" {nm}", value=float(v), format="%.4f", step=0.0005 if k == "fvf" and not oil else 0.01,
                                              key=f"vol_{fluid}_{k}_{nm}", label_visibility="collapsed")
                        for i, (nm, v) in enumerate(zip(["low", "mode", "high"], [lo, mode, hi])))
    if any(not lo <= m <= hi for lo, m, hi in spec.values()):
        st.error("Each input needs low <= most likely <= high.")
        return
    c = st.columns(3)
    corr = c[0].slider("Porosity - water saturation correlation", -0.9, 0.9, -0.3, 0.1, key="vol_corr")
    n = c[1].slider("Samples", 1000, 50000, 10000, 1000, key="vol_n")
    seed = int(c[2].number_input("Random seed", 0, 10**6, 0, 1, key="vol_seed"))
    r = vol.run(spec, n, seed, corr)
    det = vol.deterministic(spec)
    ip, rec = agg.percentiles(r["in_place"] * conv), agg.percentiles(r["recoverable"] * conv)
    st.markdown(f"**{per.capitalize()} in place ({unit})**")
    k = st.columns(5)
    for col, lab, v in zip(k, ["P90 (low)", "P50", "P10 (high)", "Mean", "Most-likely inputs"],
                           [ip["P90"], ip["P50"], ip["P10"], ip["Mean"], det["in_place"] * conv]):
        col.metric(lab, f"{v:,.1f}" if oil else f"{v:,.0f}")
    st.markdown(f"**Recoverable ({unit})**")
    k = st.columns(5)
    for col, lab, v in zip(k, ["P90 (low)", "P50", "P10 (high)", "Mean", "Most-likely inputs"],
                           [rec["P90"], rec["P50"], rec["P10"], rec["Mean"], det["recoverable"] * conv]):
        col.metric(lab, f"{v:,.1f}" if oil else f"{v:,.0f}")
    st.altair_chart(alt.Chart(pd.DataFrame({"v": r["in_place"] * conv})).mark_bar().encode(
        x=alt.X("v:Q", bin=alt.Bin(maxbins=50), title=f"In place ({unit})"), y="count():Q").properties(
        width="container", height=220, title="Distribution of in-place volume"))
    s = vol.sensitivity(r["draws"], r["in_place"]).assign(Input=lambda d: d["Input"].map(labels))
    st.altair_chart(alt.Chart(s).mark_bar().encode(
        y=alt.Y("Input:N", sort=None, title=None), x=alt.X("Rank_correlation:Q", title="Rank correlation with in place")).properties(
        width="container", height=200, title="What drives the uncertainty"))
    st.caption("The most-likely-inputs result is not the P50: skewed ranges make them differ. The recovery factor is "
               "assumed independent of the other inputs.")
    st.warning("This is a resource estimate for screening, not reserves. In place is not recoverable, and recoverable "
               "is not commercial. Inputs are only as good as your petrophysics and mapping.")
    record_box("volumetrics", {"fluid": fluid, "ranges": spec, "phi_sw_corr": corr, "n": n}, {"in_place": ip, "recoverable": rec}, seed, "volumetrics")


# END OF FILE: tabs_conv.py