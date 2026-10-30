"""GeoFlux portfolio: tab renderers."""
import altair as alt
import pandas as pd
import streamlit as st

import geoflux_runs as runs


def render_total(S):
    a, p, d = S.arith, S.prob, S.diff
    k = st.columns(4)
    k[0].metric("Portfolio EUR P90 (low)", f"{p['P90'] / 1e6:,.2f} MMBOE")
    k[1].metric("P50", f"{p['P50'] / 1e6:,.2f} MMBOE")
    k[2].metric("P10 (high)", f"{p['P10'] / 1e6:,.2f} MMBOE")
    k[3].metric("Mean", f"{p['Mean'] / 1e6:,.2f} MMBOE")
    st.markdown(f"**Two views of the same {S.n_wells} wells** (correlation {S.rho:g} between wells in the risk view)")
    st.dataframe(pd.DataFrame({
        "Case": ["P90 (low)", "P50", "P10 (high)", "Mean"],
        "Arithmetic sum (reporting convention), MMBOE": [a["P90"] / 1e6, a["P50"] / 1e6, a["P10"] / 1e6, a["Mean"] / 1e6],
        "Probabilistic portfolio (risk view), MMBOE": [p["P90"] / 1e6, p["P50"] / 1e6, p["P10"] / 1e6, p["Mean"] / 1e6],
        "Difference %": [d["P90"], d["P50"], d["P10"], 0.0]}).round(2), hide_index=True)
    st.altair_chart(alt.Chart(pd.DataFrame({"Total": S.total / 1e6})).mark_bar().encode(
        x=alt.X("Total:Q", bin=alt.Bin(maxbins=50), title="Portfolio EUR (MMBOE)"), y="count():Q").properties(
        width="container", height=200, title="Distribution of the portfolio total"))
    g = S.groups.copy()
    for c in ["P90", "P50", "P10", "Mean", "Sum_of_P90s", "Sum_of_P10s"]:
        g[c] = g[c] / 1e6
    st.markdown("**By group (MMBOE)**")
    st.dataframe(g.round(2), hide_index=True)
    st.warning("The risk view depends on the correlation you set. Wells in one basin share geology, completion design and "
               "operator, so independence (0) overstates the diversification benefit. The arithmetic view is the "
               "conservative reporting convention, not a probability statement. Means add exactly in both. Volumes only: "
               "no price, cost or NPV roll-up yet.")


def render_corr(S):
    long = S.curve.melt("Correlation", value_vars=["P90", "P50", "P10"], var_name="Case", value_name="EUR")
    long["EUR"] = long["EUR"] / 1e6
    st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("Correlation:Q", title="Correlation between wells"), y=alt.Y("EUR:Q", title="Portfolio EUR (MMBOE)",
                                                                           scale=alt.Scale(zero=False)),
        color="Case:N").properties(width="container", height=320, title="Portfolio percentiles against assumed correlation"))
    st.caption("At correlation 0 the wells' ups and downs largely cancel, so the low case is far above the sum of the wells' "
               "low cases. As correlation approaches 1 the portfolio low case falls to that sum. The high case can behave less "
               "smoothly for skewed distributions. Choose the correlation you can defend, and show this chart with it.")
    st.dataframe(S.curve.assign(**{c: S.curve[c] / 1e6 for c in ["P90", "P50", "P10", "Mean"]}).round(3), hide_index=True)


def render_wells(S):
    st.markdown("**Development plan used**")
    st.dataframe(S.plan, hide_index=True)
    st.markdown("**Each planned well (EUR, MBOE)**")
    w = S.well_tbl.copy()
    for c in ["P90", "P50", "P10", "Mean"]:
        w[c] = w[c] / 1e3
    st.dataframe(w.round(0), hide_index=True)
    st.caption("Every planned well is an independent bootstrap draw from the analog wells of its formation, scaled to its "
               "lateral length. Differences between wells here are analog scatter only.")


def render_record(S):
    st.write(f"Engine version **{S.rec['engine_version']}**, seed **{S.rec['seed']}**.")
    st.code(S.rec["inputs_sha256"], language=None)
    st.download_button("Download run record (JSON)", runs.record_json(S.rec), "geoflux_portfolio_record.json", "application/json")


# END OF FILE: tabs_port.py