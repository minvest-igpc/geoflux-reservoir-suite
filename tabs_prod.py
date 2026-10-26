"""GeoFlux oil & gas: SRV & geometry, Production & economics, and Export tabs."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import geoflux_dca as dca
import geoflux_frac as frac
import geoflux_mine as mine
from tabs_seis import csv_bytes


def render_srv(S):
    cat, wells, faults, have_stages, red, voxel, min_ev, fault_thr = S.cat, S.wells, S.faults, S.have_stages, S.red, S.voxel, S.min_ev, S.fault_thr
    xyz = cat[["X", "Y", "Z"]].to_numpy(dtype=float)
    if len(xyz) < 20:
        st.info("Not enough events for an SRV estimate.")
    else:
        hull = frac.srv_hull(xyz)
        vox = frac.srv_voxel(xyz, voxel, min_ev)
        sens = frac.srv_sensitivity(xyz, voxel, min_ev)
        m = st.columns(3)
        m[0].metric("Voxel SRV (chosen settings)", f"{vox / 1e6:.2f} million m3")
        m[1].metric("Method range", f"{sens['SRV_m3'].min() / 1e6:.2f} to {sens['SRV_m3'].max() / 1e6:.2f} million m3")
        m[2].metric("Convex hull (upper bound)", f"{hull / 1e6:.2f} million m3" if hull else "n/a")
        st.caption("The range comes from varying the voxel size (-25% to +50%) and the minimum events per "
                   "voxel (+/-1). It shows how much the answer depends on method choices; it is not a "
                   "statistical confidence interval. The hull encloses every event, outliers included, "
                   "so a large gap between hull and voxel SRV means a sparse or scattered cloud.")
        pivot = sens.assign(SRV=(sens["SRV_m3"] / 1e6).round(2)).pivot(
            index="Voxel_m", columns="Min_events", values="SRV")
        pivot.index = [f"{v:g} m" for v in pivot.index]
        pivot.columns = [f"min {c}" for c in pivot.columns]
        st.dataframe(pivot)
    color_opts = ["Magnitude"] + (["Stage"] if have_stages else [])
    color_by = st.radio("Colour events by", color_opts, horizontal=True)
    plot = cat if len(cat) <= 20000 else cat.sample(20000, random_state=0)
    enc = (alt.Color("Stage:N", sort=None) if color_by == "Stage"
           else alt.Color("Magnitude:Q", scale=alt.Scale(scheme="viridis")))

    def view(xf, yf, title, flip=False):
        pts = alt.Chart(plot).mark_circle(size=14, opacity=0.5).encode(
            x=alt.X(f"{xf}:Q", scale=alt.Scale(zero=False)),
            y=alt.Y(f"{yf}:Q", scale=alt.Scale(zero=False, reverse=flip)),
            color=enc, tooltip=["Time:T", "Magnitude:Q"])
        layers = [pts]
        for df_, col in [(wells, "black"), (faults, "#d62728")]:
            if df_ is not None:
                layers.append(alt.Chart(df_).mark_line(color=col, strokeWidth=2).encode(
                    x=f"{xf}:Q", y=alt.Y(f"{yf}:Q", scale=alt.Scale(zero=False, reverse=flip)), detail="Name:N"))
        return alt.layer(*layers).properties(title=title, width="container", height=340)

    g1, g2 = st.columns(2)
    g1.altair_chart(view("X", "Y", "Plan view (X-Y)"))
    g2.altair_chart(view("X", "Z", "Section view (X-Z), Z as depth increases downward", flip=True))
    st.caption("Black = wellbore, red = faults. Z is plotted with depth increasing downward; if your Z is "
               "elevation (up positive) the section appears upside down.")
    if faults is not None:
        near = cat[cat["Dist_to_fault_m"] <= fault_thr]
        st.write(f"**{len(near):,}** of {len(cat):,} events ({len(near) / len(cat):.0%}) lie within "
                 f"{fault_thr} m of a mapped fault.")
        if wells is not None:
            d_w, _ = mine.distance_to_workings(frac.densify(wells, 5.0), faults)
            st.write(f"Closest approach of the wellbore to a mapped fault: **{d_w.min():.0f} m**.")
        if have_stages and len(near):
            per = near["Stage"].replace("", "unassigned").value_counts().rename("Events near fault")
            st.dataframe(per.to_frame())
        st.caption("Distance to mapped faults only screens for reactivation risk. Unmapped faults are not "
                   "covered, and location error blurs the distances.")


def render_production(S):
    prod, red, skip_n, b_lo, b_hi, dmin_pct, horizon_y, n_mc, price, econ = S.prod, S.red, S.skip_n, S.b_lo, S.b_hi, S.dmin_pct, S.horizon_y, S.n_mc, S.price, S.econ
    t_all = dca.production_time_axis(prod["Date"])
    q_all = prod["Rate_boed"].to_numpy()
    fit = dca.fit_arps(t_all[skip_n:], q_all[skip_n:], (b_lo, b_hi))
    if fit is None:
        st.error("The decline-curve fit failed. Reduce the points ignored or widen the b range.")
    else:
        rng = np.random.default_rng(0)
        draws = dca.sample_arps_params(fit, n_mc, rng, (b_lo, b_hi))
        months = int(round(horizon_y * 12))
        R = np.array([dca.forecast_monthly(q, d, b, dmin_pct, horizon_y) for q, d, b in draws])
        best = dca.forecast_monthly(fit["qi"], fit["di"], fit["b"], dmin_pct, horizon_y)
        eco = dca.economics(best, price, **econ)
        mc_res = [dca.economics(r, price, **econ) for r in R]
        eur = np.array([e["eur_boe"] for e in mc_res])
        npv = np.array([e["npv"] for e in mc_res])
        be = dca.breakeven_price(best, **{k: v for k, v in econ.items()})
        p_eur, p_npv = dca.spe_percentiles(eur), dca.spe_percentiles(npv)

        k = st.columns(5)
        k[0].metric("EUR (best fit)", f"{eco['eur_boe'] / 1e3:,.0f} MBOE")
        k[1].metric("NPV (best fit)", f"{eco['npv'] / 1e6:,.2f} million")
        k[2].metric("Payback", f"{eco['payback_months']} months" if eco["payback_months"] else "none")
        k[3].metric("IRR", f"{eco['irr_annual'] * 100:.0f} %" if eco["irr_annual"] is not None else "n/a")
        k[4].metric("Breakeven price", f"{be:.1f} per BOE" if be is not None else "above range")
        r2 = fit["r2"]
        st.caption(f"Arps fit: qi = {fit['qi']:,.0f} BOE/d, Di = {fit['di']:.2f} /yr (nominal), "
                   f"b = {fit['b']:.2f}, R2 = {r2:.3f} on log rate ({fit['n']} points).")
        if fit["b_at_bound"]:
            st.warning("The fitted b sits on the edge of the allowed range, so the data cannot pin it "
                       "down. Treat long-term volumes as very uncertain.")
        if fit["n"] < 12:
            st.warning("Fewer than 12 production points: long-term forecasts are dominated by assumption.")

        yrs = (np.arange(months) + 0.5) / 12.0
        band = pd.DataFrame({"Years": yrs, "Low": np.percentile(R, 10, axis=0),
                             "High": np.percentile(R, 90, axis=0), "Best fit": best})
        hist = pd.DataFrame({"Years": t_all, "Rate": q_all})
        b0 = alt.Chart(band).encode(x=alt.X("Years:Q", title="Years since first production"))
        y_scale = alt.Scale(type="log")
        chart = (b0.mark_area(opacity=0.25).encode(y=alt.Y("Low:Q", scale=y_scale, title="BOE per day"), y2="High:Q")
                 + b0.mark_line().encode(y=alt.Y("Best fit:Q", scale=y_scale))
                 + alt.Chart(hist).mark_circle(color="#d62728", size=40).encode(x="Years:Q", y=alt.Y("Rate:Q", scale=y_scale)))
        st.altair_chart(chart.properties(width="container", height=320,
                        title="Decline forecast: history (red), best fit (line), 10th to 90th percentile (band)"))

        st.markdown("**Ranges (SPE convention: P90 is the low case, P10 the high case)**")
        rng_tab = pd.DataFrame({"": ["EUR (MBOE)", "NPV (millions)"],
                                **{k_: [p_eur[k_] / 1e3, p_npv[k_] / 1e6] for k_ in p_eur}}).round(1)
        st.dataframe(rng_tab, hide_index=True)
        hist_npv = pd.DataFrame({"NPV (millions)": npv / 1e6})
        st.altair_chart(alt.Chart(hist_npv).mark_bar().encode(
            x=alt.X("NPV (millions):Q", bin=alt.Bin(maxbins=30)), y="count():Q").properties(
            width="container", height=200, title="NPV distribution across fitted-parameter uncertainty"))
        st.warning("The ranges only reflect uncertainty in the decline-curve fit to the data you gave. "
                   "They do not include price, cost, geological or operational uncertainty, and short "
                   "histories extrapolate poorly. No reservoir properties are modelled: this is "
                   "decline-curve analysis, not a reservoir simulation or reserves estimate.")
    if fit is None:
        return None
    return {"fit": fit, "months": months, "band": band}


def render_export(S, out):
    summ, cat = S.summ, S.cat
    if summ is not None:
        st.download_button("Download stage summary (CSV)", csv_bytes(summ), "stage_summary.csv", "text/csv")
    st.download_button("Download events (CSV)", csv_bytes(cat), "events_with_stage.csv", "text/csv")
    if out is not None:
        band, months = out["band"], out["months"]
        fc = pd.DataFrame({"Month": np.arange(1, months + 1), "Rate_low_boed": band["Low"],
                           "Rate_best_boed": band["Best fit"], "Rate_high_boed": band["High"]})
        st.download_button("Download forecast (CSV)", csv_bytes(fc), "forecast.csv", "text/csv")
    st.dataframe(cat.head(20))


# END OF FILE: tabs_prod.py