"""GeoFlux type curves: new-well forecast, calibration and record tabs."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import geoflux_agg as agg
import geoflux_runs as runs


def render_new_well(S):
    if S.samples is None:
        st.info("Need at least 5 accepted analog wells for a new-well forecast.")
        return
    p = agg.percentiles(S.samples["eur"])
    st.caption(f"A new well of {S.lateral_new:,.0f} m drawn from {len(S.fits)} analog wells ({len(S.samples['eur'])} samples, "
               f"length exponent {S.beta:g}, seed {S.seed}).")
    k = st.columns(4)
    k[0].metric("EUR P90 (low)", f"{p['P90'] / 1e3:,.0f} MBOE")
    k[1].metric("EUR P50", f"{p['P50'] / 1e3:,.0f} MBOE")
    k[2].metric("EUR P10 (high)", f"{p['P10'] / 1e3:,.0f} MBOE")
    k[3].metric("EUR mean", f"{p['Mean'] / 1e3:,.0f} MBOE")
    R = S.samples["rates"]
    yrs = (np.arange(R.shape[1]) + 0.5) / 12.0
    band = pd.DataFrame({"Years": yrs, "Low": np.percentile(R, 10, axis=0), "Mid": np.percentile(R, 50, axis=0),
                         "High": np.percentile(R, 90, axis=0)})
    y = alt.Scale(type="log")
    b0 = alt.Chart(band).encode(x=alt.X("Years:Q", title="Years since first production"))
    st.altair_chart((b0.mark_area(opacity=0.25).encode(y=alt.Y("Low:Q", scale=y, title="BOE per day"), y2="High:Q")
                     + b0.mark_line().encode(y=alt.Y("Mid:Q", scale=y))).properties(
        width="container", height=300, title="New-well forecast: median (line) with P90 to P10 band"))
    st.altair_chart(alt.Chart(pd.DataFrame({"EUR_MBOE": S.samples["eur"] / 1e3})).mark_bar().encode(
        x=alt.X("EUR_MBOE:Q", bin=alt.Bin(maxbins=40), title="EUR (MBOE)"), y="count():Q").properties(
        width="container", height=180))
    st.warning("The new well is treated as a random draw from the analogs, scaled by lateral length. That assumes the same "
               "formation and a similar completion design, and it ignores differences in vintage, spacing and parent-child "
               "effects. It is a screening forecast, not a reserves estimate.")


def render_quality(S):
    cov, df = S.coverage
    c = st.columns(2)
    c[0].metric("Calibration: wells inside their predicted P90-P10 range", f"{cov * 100:.0f} %" if np.isfinite(cov) else "n/a")
    c[1].metric("Expected for an 80% range", "about 80 %")
    st.caption("Leave-one-out: each analog's EUR range is predicted from the other wells and checked against its actual EUR. "
               "Much below 80% means the ranges are too narrow (overconfident); much above means too wide.")
    if np.isfinite(cov) and not 0.6 <= cov <= 0.95:
        st.warning("Calibration is off. Treat the P90 to P10 ranges with caution, and check the analog selection.")
    if len(S.fits) < 10:
        st.warning("Fewer than 10 analog wells: the percentiles are very uncertain.")
    with st.expander("Per-well calibration table"):
        st.dataframe(df.round(0), hide_index=True)
    if len(S.rejected):
        st.markdown("**Wells rejected from fitting**")
        st.dataframe(S.rejected, hide_index=True)
    st.markdown("**Known limitations**")
    st.markdown("- Completion design, spacing and vintage are not adjusted for. Mixing them widens the ranges or biases them.\n"
                "- Recent wells have short histories, so their EUR depends on the fitted model's extrapolation.\n"
                "- Survivorship: only wells you supply are used. Missing poor wells would flatter the result.")


def render_record(S):
    st.write(f"Engine version **{S.rec['engine_version']}**, seed **{S.rec['seed']}**.")
    st.code(S.rec["inputs_sha256"], language=None)
    st.download_button("Download run record (JSON)", runs.record_json(S.rec), "geoflux_typecurve_record.json", "application/json")
    with st.expander("Show the record"):
        st.json(S.rec["outputs"])


# END OF FILE: tabs_tc2.py