"""GeoFlux type curves: well-data sidebar, cached fitting and the first tabs."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import geoflux_models as gm
import geoflux_typecurve as tc
import geoflux_wells as wl


@st.cache_data(show_spinner=False)
def demo_wells():
    return wl.make_synthetic_wells()


@st.cache_data(show_spinner="Fitting every well...")
def fit_all(series, model, terminal, years, qlim, min_points, min_r2):
    return tc.fit_wells(series, model, terminal, years, qlim, min_points, min_r2)


def load_wells_sidebar():
    """Returns (production, metadata, is_demo, notes, source label)."""
    st.sidebar.header("Well data")
    pf = st.sidebar.file_uploader("Production, all wells (.csv)", type=["csv", "txt"], help="Well, Date, Rate_boed (BOE/d).")
    mf = st.sidebar.file_uploader("Well metadata (.csv, optional)", type=["csv", "txt"],
                                  help="Well, Formation, Lateral_m (or Lateral_ft), Completion_Year, Well_Type.")
    with st.sidebar.expander("Download CSV templates"):
        dp, dm = demo_wells()
        st.download_button("Production template", dp.head(200).to_csv(index=False).encode(), "wells_production_template.csv", "text/csv")
        st.download_button("Metadata template", dm.to_csv(index=False).encode(), "wells_metadata_template.csv", "text/csv")
    if pf is None:
        dp, dm = demo_wells()
        return dp, dm, True, [], "SYNTHETIC DEMONSTRATION BASIN (36 invented wells, not real)"
    prod, err, notes = wl.load_well_production(pf)
    if err:
        st.error(f"Production upload failed: {err}")
        st.stop()
    if mf is None:
        meta = pd.DataFrame({"Well": sorted(prod["Well"].unique())})
        notes.append("No metadata: wells are treated as one group with no lateral-length scaling.")
    else:
        meta, err, w = wl.load_well_meta(mf)
        if err:
            st.error(f"Metadata upload failed: {err}")
            st.stop()
        notes += w
    return prod, meta, False, notes, f"UPLOADED: {pf.name} ({prod['Well'].nunique()} wells)"


def render_analogs(S):
    c = st.columns(4)
    c[0].metric("Wells in data", f"{S.n_total}")
    c[1].metric("Analogs selected", f"{S.n_selected}")
    c[2].metric("Fitted and accepted", f"{len(S.fits)}")
    c[3].metric("Rejected", f"{len(S.rejected)}")
    tab = S.well_tab.sort_values("EUR_MBOE", ascending=False)
    st.dataframe(tab.round({"R2": 3, "EUR_MBOE": 0, "Lateral_m": 0, "EUR_MBOE_per_km": 0}), hide_index=True)
    if "Lateral_m" in tab:
        st.altair_chart(alt.Chart(tab).mark_circle(size=70).encode(
            x=alt.X("Lateral_m:Q", title="Lateral length (m)", scale=alt.Scale(zero=False)),
            y=alt.Y("EUR_MBOE:Q", title="EUR (MBOE)"), color="Formation:N" if "Formation" in tab else alt.value("#1f77b4"),
            tooltip=list(tab.columns)).properties(width="container", height=300, title="EUR against lateral length"))
        st.caption("Volumes are scaled to a new well's lateral length with the exponent you set. A straight proportional "
                   "relationship is an assumption: check it against this chart.")
    st.altair_chart(alt.Chart(S.pc).mark_bar().encode(x=alt.X("Month:Q", title="Months on production"), y=alt.Y("N:Q", title="Wells reporting"))
                    .properties(width="container", height=160, title="How many wells inform each month"))


def render_pcurves(S):
    unit = "BOE/d per 1000 m" if S.normalised else "BOE/d per well"
    long = S.pc.melt("Month", value_vars=["P90", "P50", "P10", "Mean"], var_name="Curve", value_name="Rate")
    y = alt.Scale(type="log")
    lines = alt.Chart(long).mark_line().encode(x=alt.X("Month:Q", title="Months on production"),
                                               y=alt.Y("Rate:Q", scale=y, title=unit), color="Curve:N")
    layers = [lines]
    if S.tc_fits.get("P50"):
        yrs = np.arange(0, 120) + 0.5
        ex = pd.DataFrame({"Month": yrs, "Rate": gm.rate(S.model, yrs / 12.0, S.tc_fits["P50"]["params"], S.terminal)})
        layers.append(alt.Chart(ex).mark_line(strokeDash=[5, 4], color="black").encode(x="Month:Q", y=alt.Y("Rate:Q", scale=y)))
    st.altair_chart(alt.layer(*layers).properties(width="container", height=320,
                    title="Monthly percentile curves of the analog wells (dashed = fitted P50)"))
    rows = [{"Curve": k, "EUR_MBOE": gm.eur_boe(gm.forecast_monthly(S.model, f["params"], S.years, S.terminal), S.qlim) / 1e3,
             "R2": f["r2"]} for k, f in S.tc_fits.items() if f]
    st.dataframe(pd.DataFrame(rows).round({"EUR_MBOE": 0, "R2": 3}), hide_index=True)
    st.warning("Each month's percentile comes from a different mix of wells, and late months rely on fewer, older wells. "
               "A 'P90 curve' is not the P90 of well outcomes. For a distribution of EUR use the New well tab. "
               "Only the Mean curve adds up correctly across wells.")


# END OF FILE: tabs_tc.py