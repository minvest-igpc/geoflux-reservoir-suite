"""GeoFlux economics & risk: tab renderers."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import geoflux_runs as runs


def render_summary(S):
    sm, ep, e = S.summ, S.eur_p, S.eco
    k = st.columns(4)
    k[0].metric("NPV (best-fit case)", f"{e['npv'] / 1e6:,.2f} M")
    k[1].metric("NPV expected (mean)", f"{sm['Mean'] / 1e6:,.2f} M")
    k[2].metric("Chance NPV > 0", f"{sm['P_positive'] * 100:.0f} %")
    k[3].metric("Breakeven oil price", f"{S.be_oil:,.1f}" if S.be_oil is not None else "above range")
    k = st.columns(4)
    k[0].metric("NPV P90 (low)", f"{sm['P90'] / 1e6:,.2f} M")
    k[1].metric("NPV P50", f"{sm['P50'] / 1e6:,.2f} M")
    k[2].metric("NPV P10 (high)", f"{sm['P10'] / 1e6:,.2f} M")
    k[3].metric("IRR / payback", f"{e['irr_annual'] * 100:.0f} % / {e['payback_months']} mo"
                if e["irr_annual"] is not None and e["payback_months"] else "n/a")
    st.caption(f"EUR (decline uncertainty only): P90 {ep['P90'] / 1e3:,.0f}, P50 {ep['P50'] / 1e3:,.0f}, "
               f"P10 {ep['P10'] / 1e3:,.0f} MBOE. P90 is the low case (exceeded with 90% probability).")
    h = pd.DataFrame({"NPV (millions)": S.mc["npv"] / 1e6})
    bars = alt.Chart(h).mark_bar().encode(x=alt.X("NPV (millions):Q", bin=alt.Bin(maxbins=40)), y="count():Q")
    zero = alt.Chart(pd.DataFrame({"z": [0.0]})).mark_rule(color="#d62728", strokeWidth=2).encode(x="z:Q")
    st.altair_chart((bars + zero).properties(width="container", height=260, title="NPV distribution (red line = zero)"))
    st.warning("What these ranges include: decline-fit uncertainty within ONE model, plus the price, cost and "
               "capex ranges you set (oil and gas prices correlated). What they leave out: model choice, "
               "geology, drilling schedule and cost overruns, fiscal changes, and tax-loss carry-forward. "
               "Tax is simplified. This is a screening estimate, not a valuation or a reserves report.")


def render_tornado(S):
    df = S.tor_df.assign(Low_M=S.tor_df["NPV_at_low"] / 1e6, High_M=S.tor_df["NPV_at_high"] / 1e6,
                         Swing_M=S.tor_df["Swing"] / 1e6)
    bars = alt.Chart(df).mark_bar().encode(
        y=alt.Y("Driver:N", sort=None, title=None), x=alt.X("Low_M:Q", title="NPV (millions)"), x2="High_M:Q",
        tooltip=["Driver", "Low_setting", "High_setting", "Low_M", "High_M"])
    base = alt.Chart(pd.DataFrame({"b": [S.tor_base / 1e6]})).mark_rule(color="#d62728").encode(x="b:Q")
    st.altair_chart((bars + base).properties(width="container", height=260,
                    title="Which assumption moves NPV most (red line = base case)"))
    st.dataframe(df[["Driver", "Low_setting", "High_setting", "Low_M", "High_M", "Swing_M"]].round(2), hide_index=True)
    st.caption("Each driver is moved to its low and high setting with everything else at base. The bar spans the "
               "two resulting NPVs, so the longest bar matters most. Decline uses the simulated curves closest "
               "to the P90 and P10 EUR.")


def render_price(S):
    st.altair_chart((alt.Chart(S.curve_df.assign(NPV_M=S.curve_df["NPV"] / 1e6)).mark_line().encode(
        x=alt.X("Oil_price:Q", title="Oil price (gas moves proportionally)"),
        y=alt.Y("NPV_M:Q", title="NPV (millions)"), color="Case:N")
        + alt.Chart(pd.DataFrame({"z": [0.0]})).mark_rule(color="gray", strokeDash=[4, 3]).encode(y="z:Q")
        ).properties(width="container", height=320, title="NPV against price for three decline cases"))
    st.dataframe(pd.DataFrame({"Case": list(S.be_cases), "Breakeven oil price": [v if v is not None else np.nan
                                                                                for v in S.be_cases.values()]}).round(1),
                 hide_index=True)
    st.caption("Breakeven is the oil price at which NPV is zero, with the gas price scaled in proportion.")


def render_cashflow(S):
    cf = S.eco["cf"]
    cum = -S.deck["capex"] + np.cumsum(cf)
    df = pd.DataFrame({"Month": np.arange(1, len(cf) + 1), "Cumulative_M": cum / 1e6})
    st.altair_chart((alt.Chart(df).mark_line().encode(x="Month:Q", y=alt.Y("Cumulative_M:Q", title="Cumulative cash (millions)"))
                     + alt.Chart(pd.DataFrame({"z": [0.0]})).mark_rule(color="#d62728").encode(y="z:Q")).properties(
        width="container", height=280, title="Cumulative cash flow, best-fit case (undiscounted)"))
    e = S.eco
    st.write(f"Producing life **{e['life_months']} months**; total revenue **{e['revenue_total'] / 1e6:,.1f} M**; "
             f"operating cash flow **{e['opcf_total'] / 1e6:,.1f} M**.")


def render_record(S):
    st.dataframe(pd.DataFrame({"Input": list(S.deck), "Value": [str(v) for v in S.deck.values()]}), hide_index=True)
    st.dataframe(pd.DataFrame([{"Input": k, "Low": v[0], "Base": v[1], "High": v[2]} for k, v in S.spec.items()]).round(2),
                 hide_index=True)
    st.write(f"Engine version **{S.rec['engine_version']}**, seed **{S.rec['seed']}**.")
    st.code(S.rec["inputs_sha256"], language=None)
    st.download_button("Download run record (JSON)", runs.record_json(S.rec), "geoflux_econ_record.json", "application/json")


# END OF FILE: tabs_econ.py