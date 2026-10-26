"""GeoFlux oil & gas: Data & QC, Stages & seismicity, and Traffic light tabs."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import geoflux_frac as frac


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def fmt(v, spec=",.0f", none="n/a"):
    return none if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)


def render_qc(S):
    cat, stages, have_stages, mc, b_res, seis_ok, lag_min, voxel = S.cat, S.stages, S.have_stages, S.mc, S.b_res, S.seis_ok, S.lag_min, S.voxel
    c = st.columns(6)
    c[0].metric("Events", f"{len(cat):,}")
    c[1].metric("Span", f"{(cat['Time'].max() - cat['Time'].min()).total_seconds() / 86400:.1f} days")
    c[2].metric("Mc", fmt(mc, ".2f"))
    c[3].metric("b-value", f"{b_res['b']:.2f} +/- {b_res['sigma']:.2f}" if seis_ok else "n/a")
    c[4].metric("Stages", f"{len(stages)}" if have_stages else "no log")
    c[5].metric("Median loc. error", f"{cat['Loc_Error_m'].median():.0f} m" if "Loc_Error_m" in cat else "n/a")
    if not seis_ok:
        st.warning("Too few events to estimate Mc and b-value (need about 50 events above Mc).")
    if have_stages:
        assigned = float((cat["Stage"] != "").mean())
        st.write(f"**{assigned:.0%}** of events fall inside a stage pumping window (plus the "
                 f"{lag_min}-minute lag).")
        if assigned < 0.5:
            st.warning("Fewer than half of the events match a stage. Check that the catalogue and the "
                       "treatment log use the same clock and timezone.")
    else:
        st.info("No treatment stage log: stage-based tabs are limited. Upload one to link seismicity "
                "to pumping.")
    if "Loc_Error_m" in cat and voxel < cat["Loc_Error_m"].median():
        st.warning("The voxel size is smaller than the median location error, so the SRV is not "
                   "resolved at that scale. Increase the voxel size.")
    mags = cat["Magnitude"].to_numpy()
    grid = np.linspace(mags.min(), mags.max(), 30)
    cum = pd.DataFrame({"M": grid, "log10_N": [np.log10(max((mags >= g).sum(), 1)) for g in grid]})
    st.altair_chart(alt.Chart(cum).mark_line(point=True).encode(
        x=alt.X("M:Q", title="Magnitude"), y=alt.Y("log10_N:Q", title="log10 N(>=M)")).properties(
        width="container", height=260, title="Frequency-magnitude distribution"))


def render_stages(S):
    stages, mc, b_res, summ, G_gpa, amber, red, next_vol = S.stages, S.mc, S.b_res, S.summ, S.G_gpa, S.amber, S.red, S.next_vol
    if summ is None:
        st.info("Needs a stage log and enough events for Mc and b-value.")
    else:
        sig_total = frac.seismogenic_index(int(summ["Events_above_Mc"].sum()),
                                           float(summ["Volume_m3"].sum()), b_res["b"], mc)
        k = st.columns(3)
        k[0].metric("Total injected volume", f"{summ['Volume_m3'].sum():,.0f} m3")
        k[1].metric("Pooled seismogenic index", fmt(sig_total, ".2f"))
        k[2].metric("Largest event", f"M {summ['Mmax'].max():.1f}")
        show = summ.copy()
        show["Sigma_err"] = show["Sigma_err"].round(2)
        st.dataframe(show.round({"Volume_m3": 0, "Mmax": 1, "Sigma": 2, "Cum_moment_Nm": 0,
                                 "McGarr_Mw": 2, "McGarr_utilisation": 4}), hide_index=True)
        s1, s2 = st.columns(2)
        with s1:
            base = alt.Chart(summ.assign(lo=summ["Sigma"] - summ["Sigma_err"],
                                         hi=summ["Sigma"] + summ["Sigma_err"]))
            st.altair_chart((base.mark_rule().encode(x=alt.X("Stage:O", sort=None), y="lo:Q", y2="hi:Q")
                             + base.mark_circle(size=70).encode(
                                 x=alt.X("Stage:O", sort=None),
                                 y=alt.Y("Sigma:Q", title="Seismogenic index", scale=alt.Scale(zero=False)),
                                 tooltip=["Stage", "Sigma", "Events_above_Mc", "Volume_m3"]))
                            .properties(width="container", height=280,
                                        title="Seismogenic index by stage (+/-1 sigma)"))
            st.caption("A stage standing well above the others produces more seismicity per unit of "
                       "fluid. Look at where it is.")
        with s2:
            cv = np.cumsum(summ["Volume_m3"].to_numpy())
            cm = np.cumsum(summ["Cum_moment_Nm"].to_numpy())
            d = pd.DataFrame({"Volume_m3": cv, "Observed": np.maximum(cm, 1.0),
                              "McGarr bound": G_gpa * 1e9 * cv})
            long = d.melt("Volume_m3", var_name="Series", value_name="Moment_Nm")
            st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
                x=alt.X("Volume_m3:Q", title="Cumulative injected volume (m3)",
                        scale=alt.Scale(type="log", zero=False)),
                y=alt.Y("Moment_Nm:Q", title="Cumulative seismic moment (N m)", scale=alt.Scale(type="log")),
                color="Series:N").properties(width="container", height=280,
                                             title="Seismic moment versus injected volume"))
            st.caption("McGarr (2014): the largest moment is bounded by G times the net injected "
                       "volume. It is a physical-limit argument, not a guarantee; exceptions exist.")
        st.markdown("**Look-ahead: chance of exceeding a magnitude when pumping the next stage**")
        sig_worst = float(summ["Sigma"].max())
        rows = []
        for m_star in sorted({0.0, 1.0, amber, red}):
            rows.append({"Magnitude": m_star,
                         "P (pooled index)": frac.exceedance_from_index(sig_total, b_res["b"], next_vol, m_star, mc),
                         "P (worst stage so far)": frac.exceedance_from_index(sig_worst, b_res["b"], next_vol, m_star, mc)})
        la = pd.DataFrame(rows)
        for col in ["P (pooled index)", "P (worst stage so far)"]:
            la[col] = (la[col] * 100).round(1).astype(str) + " %"
        st.dataframe(la, hide_index=True)
        st.caption(f"For a {next_vol:,.0f} m3 stage. Assumes a Poisson process and a seismogenic index "
                   "that does not change between stages. A stage near a fault can differ a lot.")


def render_traffic(S):
    cat, have_stages, summ, amber, red = S.cat, S.have_stages, S.summ, S.amber, S.red
    big = cat[cat["Magnitude"] >= amber].copy()
    overall = frac.traffic_light(float(cat["Magnitude"].max()), amber, red)
    colour = {"Green": "#2ca02c", "Amber": "#ff9f1c", "Red": "#d62728"}[overall]
    st.markdown(f"<div style='padding:12px;border-radius:8px;background:{colour};color:white;"
                f"font-size:20px'>Overall status: <b>{overall}</b> (largest event M "
                f"{cat['Magnitude'].max():.1f}; amber at {amber:g}, red at {red:g})</div>",
                unsafe_allow_html=True)
    ev_plot = cat if len(cat) <= 20000 else cat.sample(20000, random_state=0)
    pts = alt.Chart(ev_plot).mark_circle(size=18, opacity=0.5).encode(
        x="Time:T", y=alt.Y("Magnitude:Q", scale=alt.Scale(zero=False)),
        tooltip=["Time:T", "Magnitude:Q"] + (["Stage:N"] if have_stages else []))
    rules = alt.Chart(pd.DataFrame({"y": [amber, red], "c": ["Amber", "Red"]})).mark_rule(
        strokeDash=[6, 3], strokeWidth=2).encode(
        y="y:Q", color=alt.Color("c:N", scale=alt.Scale(domain=["Amber", "Red"], range=["#ff9f1c", "#d62728"])))
    st.altair_chart((pts + rules).properties(width="container", height=320))
    if summ is not None:
        st.markdown("**Status by stage**")
        st.dataframe(summ[["Stage", "Volume_m3", "Mmax", "Status"]].round({"Volume_m3": 0, "Mmax": 1}),
                     hide_index=True)
    if len(big):
        st.markdown(f"**Events at or above amber (M {amber:g})**")
        cols = ["Time", "Magnitude"] + (["Stage"] if have_stages else []) + \
               (["Dist_to_fault_m"] if "Dist_to_fault_m" in big else [])
        st.dataframe(big[cols].sort_values("Magnitude", ascending=False).head(50), hide_index=True)
    st.caption("A traffic light compares magnitudes with thresholds you set. It is only as good as "
               "those thresholds, the catalogue's completeness and the monitoring network's latency.")


# END OF FILE: tabs_seis.py