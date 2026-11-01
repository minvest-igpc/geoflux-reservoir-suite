"""GeoFlux decline models: tab renderers for the Decline Models page."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import geoflux_agg as agg
import geoflux_backtest as bt
import geoflux_models as gm
import geoflux_runs as runs


def render_compare(S):
    cmp, fits, t, q = S.cmp, S.fits, S.t, S.q
    show = cmp[["Label", "R2", "Resid_sd", "AICc", "dAICc", "At_bound", "Params"]]
    st.dataframe(show.round({"R2": 4, "Resid_sd": 3, "AICc": 1, "dAICc": 1}), hide_index=True)
    st.caption("Lower AICc is better. Models within about 2 of each other cannot be told apart on this "
               "data. 'At_bound' lists parameters pinned to their limits: the data cannot determine them.")
    grid = np.linspace(float(t.min()), max(5.0, float(t.max()) * 1.5), 200)
    curves = pd.concat([pd.DataFrame({"Years": grid, "Rate": gm.rate(m, grid, f["params"], S.terminal),
                                      "Series": gm.label(m)}) for m, f in fits.items()])
    pts = alt.Chart(pd.DataFrame({"Years": t, "Rate": q})).mark_circle(color="#d62728", size=45).encode(
        x="Years:Q", y=alt.Y("Rate:Q", scale=alt.Scale(type="log")))
    lines = alt.Chart(curves).mark_line().encode(
        x=alt.X("Years:Q", title="Years since first production"),
        y=alt.Y("Rate:Q", scale=alt.Scale(type="log"), title="BOE per day"), color="Series:N")
    st.altair_chart((lines + pts).properties(width="container", height=340,
                    title="Fits to the data (red) and their extrapolation"))


def render_backtest(S):
    if S.bts.empty:
        st.info("Backtesting needs at least 12 production points.")
        return
    st.caption(f"Each model is fitted to the first k points and asked to predict the rest, for k = "
               f"{', '.join(str(k) for k in S.origins)}. 'MAE_log' is the typical error in log rate "
               "(0.10 is about 10%). A positive cumulative error means the model over-predicted volume.")
    st.dataframe(S.bts.round({"MAE_log": 3, "Cum_error_pct": 1, "Abs_cum_error_pct": 1}), hide_index=True)
    st.altair_chart(alt.Chart(S.bts).mark_bar().encode(
        x=alt.X("Label:N", sort=None, title=None), y=alt.Y("MAE_log:Q", title="Hindcast error (log rate)"),
        color="Label:N").properties(width="container", height=240))
    with st.expander("Every hindcast"):
        st.dataframe(S.bt.round(3), hide_index=True)
    st.warning("A model that hindcasts well on a short history is not proven for the long term. "
               "Treat the ranking as evidence, not as a verdict.")


def render_forecast(S):
    sim = S.sim
    p = agg.percentiles(sim["eur"])
    st.caption(f"Model: **{gm.label(S.model_sel)}**, {len(sim['eur'])} parameter samples, seed {S.seed}.")
    k = st.columns(4)
    k[0].metric("EUR P90 (low)", f"{p['P90'] / 1e3:,.0f} MBOE")
    k[1].metric("EUR P50", f"{p['P50'] / 1e3:,.0f} MBOE")
    k[2].metric("EUR P10 (high)", f"{p['P10'] / 1e3:,.0f} MBOE")
    k[3].metric("EUR mean", f"{p['Mean'] / 1e3:,.0f} MBOE")
    yrs = (np.arange(sim["rates"].shape[1]) + 0.5) / 12.0
    band = pd.DataFrame({"Years": yrs, "Low": np.percentile(sim["rates"], 10, axis=0),
                         "Mid": np.percentile(sim["rates"], 50, axis=0), "High": np.percentile(sim["rates"], 90, axis=0)})
    y = alt.Scale(type="log")
    b0 = alt.Chart(band).encode(x=alt.X("Years:Q", title="Years since first production"))
    pts = alt.Chart(pd.DataFrame({"Years": S.t, "Rate": S.q})).mark_circle(color="#d62728", size=40).encode(
        x="Years:Q", y=alt.Y("Rate:Q", scale=y))
    st.altair_chart((b0.mark_area(opacity=0.25).encode(y=alt.Y("Low:Q", scale=y, title="BOE per day"), y2="High:Q")
                     + b0.mark_line().encode(y=alt.Y("Mid:Q", scale=y)) + pts).properties(
        width="container", height=320, title="Forecast: history (red), median (line), P90 to P10 (band)"))
    st.markdown("**Model spread: best-fit EUR by model (same horizon and limit)**")
    rows = [{"Model": gm.label(m), "EUR_MBOE": gm.eur_boe(gm.forecast_monthly(m, f["params"], S.years, S.terminal), S.qlim) / 1e3}
            for m, f in S.fits.items()]
    st.dataframe(pd.DataFrame(rows).round(0), hide_index=True)
    st.warning("The P90/P50/P10 above reflect parameter uncertainty within ONE model. The spread between "
               "models in the table is often larger. Neither includes geological, price or operational "
               "uncertainty, and none of this is a reserves estimate.")


def render_diagnostics(S):
    d = bt.loglog_slope(S.t, S.q)
    st.metric("Late-time log-log slope", f"{d['slope']:.2f}" if np.isfinite(d["slope"]) else "n/a")
    st.write(d["hint"])
    df = pd.DataFrame({"inv_sqrt_t": S.t ** -0.5, "Rate": S.q})
    st.altair_chart(alt.Chart(df).mark_circle(size=45).encode(
        x=alt.X("inv_sqrt_t:Q", title="1 / sqrt(time in years)"), y=alt.Y("Rate:Q", title="BOE per day")).properties(
        width="container", height=280, title="Rate against 1/sqrt(t): a straight line suggests linear flow"))
    st.caption("These are hints only. Identifying a flow regime properly needs flowing-pressure data.")


@st.cache_data(show_spinner=False)
def _explainer(n_wells, p50, sigma, rho, seed=1, n=40_000):
    z = np.random.default_rng(seed).standard_normal((n, n_wells))
    return agg.aggregate({f"Well {i + 1}": p50 * np.exp(sigma * z[:, i]) for i in range(n_wells)}, rho)


def render_portfolio_explainer(S=None):
    st.markdown("Why percentiles do not simply add. Wells here are lognormal and identical; only the "
                "correlation between them changes.")
    c = st.columns(4)
    n_wells = c[0].slider("Wells", 2, 50, 10)
    p50 = c[1].number_input("P50 per well (MBOE)", 10.0, 5000.0, 500.0, 10.0)
    sigma = c[2].slider("Uncertainty (lognormal sigma)", 0.1, 1.0, 0.5, 0.05)
    rho = c[3].slider("Correlation between wells", 0.0, 1.0, 0.0, 0.1)
    r = _explainer(n_wells, p50, sigma, rho)
    a, pr, d = r["arithmetic"], r["probabilistic"], r["difference_pct"]
    st.dataframe(pd.DataFrame({
        "Case": ["P90 (low)", "P50", "P10 (high)", "Mean"],
        "Arithmetic sum (reporting convention)": [a["P90"], a["P50"], a["P10"], a["Mean"]],
        "Probabilistic portfolio (risk view)": [pr["P90"], pr["P50"], pr["P10"], pr["Mean"]],
        "Difference %": [d["P90"], d["P50"], d["P10"], 0.0]}).round(0), hide_index=True)
    st.altair_chart(alt.Chart(pd.DataFrame({"Total": r["total_samples"]})).mark_bar().encode(
        x=alt.X("Total:Q", bin=alt.Bin(maxbins=50), title="Portfolio total (MBOE)"), y="count():Q").properties(
        width="container", height=200, title="Distribution of the portfolio total"))
    st.caption("Reserves reporting rules generally call for arithmetic sums by category above project level; "
               "that low case is conservative and the high case optimistic. The probabilistic view is for "
               "investment-risk decisions and depends on the correlation you assume: independence overstates "
               "the benefit. Means add exactly in both.")


def render_record(S):
    rec = S.rec
    st.write(f"Engine version **{rec['engine_version']}**, method **{rec['method']}**, seed **{rec['seed']}**.")
    st.code(rec["inputs_sha256"], language=None)
    st.caption("SHA-256 of the inputs. The same inputs, method, version and seed reproduce the same results.")
    st.download_button("Download run record (JSON)", runs.record_json(rec), "geoflux_run_record.json", "application/json")
    with st.expander("Show the record"):
        st.json(rec["outputs"])


# END OF FILE: tabs_models.py