# geoflux/app.py - GeoFlux Mine Seismology Module (PROTOTYPE)
#
# Not validated. Not for safety-critical decisions. The oil & gas demo lives in
# frac_demo.py until its module is rebuilt.

import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import geoflux_core as core
import geoflux_mine as mine

st.set_page_config(page_title="GeoFlux: Mine Seismology (prototype)", layout="wide")


@st.cache_data(show_spinner=False)
def demo_data():
    cat, blasts, workings, _truth = mine.make_synthetic_mine()
    return cat, blasts, workings


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# ----------------------------------------------------------------- sidebar: data
st.sidebar.header("Data")
cat_file = st.sidebar.file_uploader("Seismic event catalogue (.csv)", type=["csv", "txt"],
    help="Required: Time, X, Y, Z (mine grid, metres) and Magnitude or Energy_J. "
         "Optional: Event_Type, Loc_Error_m, Moment_Nm.")
blast_file = st.sidebar.file_uploader("Blast log (.csv, optional)", type=["csv", "txt"],
    help="Required: Time. Optional: X, Y, Z, Notes.")
work_file = st.sidebar.file_uploader("Workings (.csv, optional)", type=["csv", "txt"],
    help="Columns: Name (optional), X, Y, Z. Rows with the same Name form a 3-D polyline "
         "in file order. Must use the same coordinate system and Z convention as the catalogue.")

load_notes = []
if cat_file is not None:
    cat, err, w = mine.load_catalogue(cat_file)
    if err:
        st.error(f"Catalogue upload failed: {err}")
        st.stop()
    load_notes += w
    blasts, workings = None, None
    if blast_file is not None:
        blasts, err, w = mine.load_blast_log(blast_file)
        if err:
            st.error(f"Blast log upload failed: {err}")
            st.stop()
        load_notes += w
    if work_file is not None:
        workings, err, w = mine.load_workings(work_file)
        if err:
            st.error(f"Workings upload failed: {err}")
            st.stop()
        load_notes += w
    is_demo = False
    source = f"UPLOADED: {cat_file.name} ({len(cat):,} events)"
else:
    cat, blasts, workings = demo_data()
    is_demo = True
    source = "SYNTHETIC DEMONSTRATION MINE (randomly generated; not a real site)"
    if blast_file is not None or work_file is not None:
        load_notes.append("Blast log / workings are ignored until a catalogue is uploaded.")

with st.sidebar.expander("Download CSV templates"):
    st.caption("Examples of the expected formats (from the demonstration mine).")
    st.download_button("Catalogue template", csv_bytes(demo_data()[0].head(200)),
                       "catalogue_template.csv", "text/csv")
    st.download_button("Blast log template", csv_bytes(demo_data()[1].head(50)),
                       "blast_log_template.csv", "text/csv")
    st.download_button("Workings template", csv_bytes(demo_data()[2]),
                       "workings_template.csv", "text/csv")

# ------------------------------------------------------------ sidebar: settings
st.sidebar.header("Blast discrimination")
blast_window_s = st.sidebar.number_input("Direct blast window after a logged blast (s)", 5, 600, 60)
post_min = st.sidebar.number_input("Post-blast response window (min)", 1, 240, 30)
hours_txt = st.sidebar.text_input("Typical blast hours (24 h, comma-separated)",
                                  value="5.5, 17.5" if is_demo else "",
                                  help="e.g. 5.5, 17.5 means 05:30 and 17:30. Leave empty if unknown.")
hour_tol = st.sidebar.number_input("Blast-hour tolerance (min)", 5, 120, 30)
try:
    blast_hours = tuple(float(x) for x in hours_txt.replace(";", ",").split(",") if x.strip())
except ValueError:
    st.sidebar.error("Blast hours must be numbers, e.g. 5.5, 17.5")
    blast_hours = ()

cat = cat.copy()
cat["Class"] = mine.classify_events(cat, blasts, blast_window_s, 5.0, post_min, blast_hours, hour_tol)
include = st.sidebar.multiselect("Classes included in analysis", mine.ALL_CLASSES,
                                 default=mine.DEFAULT_INCLUDE,
                                 help="Post-blast seismicity is real hazard, so it is kept by default.")
df = cat[cat["Class"].isin(include)].reset_index(drop=True)
if len(df) < 50:
    st.error("Fewer than 50 events are included in the analysis. Adjust the filters or the data.")
    st.stop()

st.sidebar.header("Statistics")
auto_bin = core.detect_mag_bin(df["Magnitude"])
mag_bin = st.sidebar.selectbox("Magnitude rounding", [0.0, 0.1],
                               index=1 if auto_bin == 0.1 else 0,
                               format_func=lambda v: "continuous" if v == 0 else f"rounded to {v}",
                               help="Auto-detected from the data; override if wrong.")
mc_auto = core.estimate_mc_maxc(df["Magnitude"], mag_bin)
if mc_auto is None:
    st.error("Too few events to estimate the magnitude of completeness.")
    st.stop()
mc = (st.sidebar.number_input("Mc override", value=float(mc_auto), step=0.1, format="%.2f")
      if st.sidebar.checkbox("Override Mc") else mc_auto)
bwin = st.sidebar.slider("b-value window (events)", 50, 500, 150, 25)
baseline_days = st.sidebar.slider("Rate baseline (first N days)", 7, 90, 30, 1)
alpha = st.sidebar.select_slider("Rate-flag significance", [0.05, 0.01, 0.001], value=0.01)
near_m = st.sidebar.slider("'Near excavation' distance (m)", 5, 200, 30, 5)

st.sidebar.header("Hazard screen")
recent_days = st.sidebar.slider("Recent-rate window (days)", 3, 60, 14)
horizon_days = st.sidebar.slider("Look-ahead horizon (days)", 1, 30, 7)
thr_txt = st.sidebar.text_input("Magnitude thresholds", "0, 1, 2, 3")
try:
    thresholds = tuple(float(x) for x in thr_txt.split(",") if x.strip()) or (0.0, 1.0, 2.0)
except ValueError:
    st.sidebar.error("Thresholds must be numbers, e.g. 0, 1, 2")
    thresholds = (0.0, 1.0, 2.0)

# ---------------------------------------------------------------- computations
if workings is not None:
    cat["Dist_to_working_m"], cat["Nearest_working"] = mine.distance_to_workings(cat, workings)
    df = cat[cat["Class"].isin(include)].reset_index(drop=True)
b_all = core.estimate_b_value(df["Magnitude"], mc, mag_bin)
if b_all is None:
    st.error(f"Fewer than 50 events at or above Mc = {mc:.2f}. Lower Mc or add data.")
    st.stop()
daily = mine.poisson_rate_flags(mine.daily_table(df, mc), baseline_days, alpha)
bseries = core.b_value_series(df["Time"], df["Magnitude"], mc, mag_bin, bwin, max(bwin // 5, 10))

# --------------------------------------------------------------------- header
st.title("GeoFlux: Mine Seismology Module")
st.warning("PROTOTYPE: not validated. Do not use as the basis for re-entry, support design "
           "or other safety decisions. Outputs support, and never replace, a qualified "
           "mine seismologist and your site's seismic risk management plan.")
(st.warning if is_demo else st.info)(f"DATA SOURCE: {source}")
for note in load_notes:
    st.caption(f"Data note: {note}")

tabs = st.tabs(["Data & QC", "Blast discrimination", "Spatial", "b-value & rate",
                "Hazard screen", "Export"])

# ------------------------------------------------------------------ tab 1: QC
with tabs[0]:
    c = st.columns(6)
    c[0].metric("Events (all)", f"{len(cat):,}")
    c[1].metric("In analysis", f"{len(df):,}")
    c[2].metric("Span", f"{(cat['Time'].max() - cat['Time'].min()).days} days")
    c[3].metric("Mc", f"{mc:.2f}")
    c[4].metric("b (all included)", f"{b_all['b']:.2f} +/- {b_all['sigma']:.2f}")
    c[5].metric("Median loc. error",
                f"{cat['Loc_Error_m'].median():.0f} m" if "Loc_Error_m" in cat else "n/a")
    st.caption(f"b-value from {b_all['n']:,} events at or above Mc = {mc:.2f} "
               f"(Aki maximum likelihood; magnitudes {'continuous' if mag_bin == 0 else f'rounded to {mag_bin}'}).")

    mc_month = (df.groupby(df["Time"].dt.to_period("M"))["Magnitude"]
                .apply(lambda m: core.estimate_mc_maxc(m, mag_bin) if len(m) >= 50 else np.nan))
    mc_month = mc_month.rename("Mc").reset_index()
    mc_month["Time"] = mc_month["Time"].astype(str)
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**Completeness magnitude by month**")
        if mc_month["Mc"].notna().sum() >= 2:
            st.altair_chart(alt.Chart(mc_month).mark_line(point=True).encode(
                x=alt.X("Time:N", title="Month"), y=alt.Y("Mc:Q", scale=alt.Scale(zero=False))
            ).properties(width="container", height=240))
            spread = float(mc_month["Mc"].max() - mc_month["Mc"].min())
            if spread > 0.3:
                st.warning(f"Mc varies by {spread:.1f} magnitude units across months: catalogue "
                           "completeness is changing, so rate and b-value trends may be artefacts "
                           "of network changes.")
        else:
            st.caption("Not enough events per month to estimate Mc by month.")
    with q2:
        st.markdown("**Events by hour of day (all events)**")
        hod = cat["Time"].dt.hour.value_counts().reindex(range(24), fill_value=0)
        st.altair_chart(alt.Chart(hod.rename("Events").rename_axis("Hour").reset_index())
                        .mark_bar().encode(x=alt.X("Hour:O"), y="Events:Q")
                        .properties(width="container", height=240))
        st.caption("Sharp peaks at fixed hours usually mean blasting. Compare with your blast log.")
    if "Loc_Error_m" in cat:
        poor = float((cat["Loc_Error_m"] > 50).mean())
        if poor > 0.1:
            st.warning(f"{poor:.0%} of events have location error above 50 m: distance-to-working "
                       "results will be unreliable for those events.")

# -------------------------------------------------- tab 2: blast discrimination
with tabs[1]:
    st.markdown("**How events were classified**")
    counts = cat["Class"].value_counts().reindex(mine.ALL_CLASSES, fill_value=0)
    st.dataframe(pd.DataFrame({"Events": counts, "In analysis": [k in include for k in counts.index]}))
    daily_cls = (cat.assign(Date=cat["Time"].dt.floor("D")).groupby(["Date", "Class"]).size()
                 .rename("Events").reset_index())
    st.altair_chart(alt.Chart(daily_cls).mark_bar().encode(
        x="Date:T", y="Events:Q", color="Class:N").properties(width="container", height=260))
    if blasts is None:
        st.info("No blast log supplied: only hour-of-day flags (and any Event_Type column) are used.")
    with st.expander("What this does and does not do"):
        st.markdown(
            "- Uses **timestamps only**: a logged-blast window, a post-blast window, and typical blast "
            "hours. It performs **no waveform analysis**.\n"
            "- Genuine seismicity falling inside a blast window is labelled as blast and cannot be "
            "separated without waveform features.\n"
            "- **Post-blast seismicity is kept** in the analysis by default: it is real rock-mass "
            "response, and removing it would hide hazard.\n"
            "- **Blast-hour (unconfirmed)** events are only flagged by time of day; many are genuine.\n"
            "- Accuracy depends on clock synchronisation between the seismic system and the blast log.")

# ------------------------------------------------------------------ tab 3: spatial
with tabs[2]:
    options = ["Class", "Magnitude"] + (["Distance to working (m)"] if workings is not None else [])
    color_by = st.radio("Colour events by", options, horizontal=True)
    plot = df if len(df) <= 20000 else df.sample(20000, random_state=0)
    if len(df) > 20000:
        st.caption("Showing a random sample of 20,000 events.")
    plot = plot.rename(columns={"Dist_to_working_m": "Distance to working (m)"})
    enc = {"Class": alt.Color("Class:N"), "Magnitude": alt.Color("Magnitude:Q", scale=alt.Scale(scheme="viridis")),
           "Distance to working (m)": alt.Color("Distance to working (m):Q", scale=alt.Scale(scheme="plasma"))}[color_by]

    def view(xf, yf, title):
        pts = alt.Chart(plot).mark_circle(size=22, opacity=0.6).encode(
            x=alt.X(f"{xf}:Q", scale=alt.Scale(zero=False)), y=alt.Y(f"{yf}:Q", scale=alt.Scale(zero=False)),
            color=enc, tooltip=["Time:T", "Magnitude:Q", "Class:N"])
        layers = [pts]
        if workings is not None:
            layers.append(alt.Chart(workings).mark_line(color="black", strokeWidth=2).encode(
                x=f"{xf}:Q", y=f"{yf}:Q", detail="Name:N"))
        return alt.layer(*layers).properties(title=title, width="container", height=360)

    s1, s2 = st.columns(2)
    s1.altair_chart(view("X", "Y", "Plan view (X-Y)"))
    s2.altair_chart(view("X", "Z", "Section view (X-Z)"))
    st.caption("Black lines are the supplied workings. Axes are not forced to equal scale.")
    if workings is not None:
        near = df[df["Dist_to_working_m"] <= near_m]
        st.write(f"**{len(near):,}** of {len(df):,} analysed events ({len(near) / len(df):.0%}) lie within "
                 f"{near_m} m of a working.")
        if len(near):
            top = near["Nearest_working"].value_counts().rename("Events near working").to_frame()
            st.dataframe(top)

# --------------------------------------------------------- tab 4: b-value & rate
with tabs[3]:
    st.markdown("**Moving b-value (fixed number of events per window)**")
    if len(bseries):
        band = bseries.assign(lo=bseries["b"] - bseries["sigma"], hi=bseries["b"] + bseries["sigma"])
        base = alt.Chart(band).encode(x="Time:T")
        st.altair_chart((base.mark_area(opacity=0.25).encode(
            y=alt.Y("lo:Q", title="b-value", scale=alt.Scale(zero=False)), y2="hi:Q")
            + base.mark_line().encode(y="b:Q")).properties(width="container", height=260))
        st.caption(f"Windows of {bwin} events above Mc = {mc:.2f}; band is +/-1 sigma (Aki). "
                   "A falling b-value is a possible indicator of rising stress, not a prediction.")
    else:
        st.info("Not enough events for the chosen window; reduce the window size.")
    if workings is not None:
        n_ev, f_ev = df[df["Dist_to_working_m"] <= near_m], df[df["Dist_to_working_m"] > near_m]
        bn = core.estimate_b_value(n_ev["Magnitude"], mc, mag_bin)
        bf = core.estimate_b_value(f_ev["Magnitude"], mc, mag_bin)
        k1, k2 = st.columns(2)
        k1.metric(f"b within {near_m} m of workings", f"{bn['b']:.2f} +/- {bn['sigma']:.2f}" if bn else "n/a (too few)")
        k2.metric("b elsewhere", f"{bf['b']:.2f} +/- {bf['sigma']:.2f}" if bf else "n/a (too few)")

    st.markdown("**Daily event rate (events at or above Mc)**")
    rate_chart = alt.Chart(daily).mark_bar().encode(
        x="Date:T", y=alt.Y("Count:Q", title="Events per day"),
        color=alt.condition("datum.Flag", alt.value("#d62728"), alt.value("#7f7f7f")),
        tooltip=["Date:T", "Count:Q", alt.Tooltip("p_value:Q", format=".4f")])
    st.altair_chart(rate_chart.properties(width="container", height=260))
    nflag = int(daily["Flag"].sum())
    st.caption(f"Baseline rate {daily['Baseline_rate'].iloc[0]:.1f} events/day from the first "
               f"{baseline_days} days. {nflag} day(s) flagged (red) at p < {alpha}. Seismicity is clustered, "
               "so some flags will be false alarms; a flag means 'review this day', not 'hazard'.")
    if "Energy_J" in daily:
        cum = daily.assign(Cumulative_energy_J=daily["Energy_J"].cumsum())
        st.markdown("**Cumulative radiated energy (J)**")
        st.altair_chart(alt.Chart(cum).mark_line().encode(
            x="Date:T", y=alt.Y("Cumulative_energy_J:Q", title="Joules", scale=alt.Scale(type="log"))
        ).properties(width="container", height=220))

# ------------------------------------------------------------- tab 5: hazard
with tabs[4]:
    tab, ctx = mine.hazard_screen(df, mc, b_all, recent_days, horizon_days, thresholds)
    h = st.columns(3)
    h[0].metric(f"Events at or above Mc, last {recent_days} d", f"{ctx['recent_events']}")
    h[1].metric("Recent rate", f"{ctx['rate_per_day']:.1f} / day")
    h[2].metric("Largest event in catalogue", f"M {ctx['largest']:.1f}")
    shown = tab.copy()
    for col in ["P(>= threshold)", "Low", "High"]:
        shown[col] = (shown[col] * 100).round(1).astype(str) + " %"
    shown.insert(0, "Window", f"next {horizon_days} d")
    st.dataframe(shown, hide_index=True)
    st.warning("Statistical screening only. Assumes a stationary Poisson process with a "
               "Gutenberg-Richter magnitude law. Clustering (aftershocks, swarms, blast response) breaks "
               "the Poisson assumption and can make true probability higher than shown. Low/High use "
               "rate +/- sqrt(N) and b -/+ sigma and are not confidence intervals. This is not a rockburst forecast.")
    st.caption("Not included: ground motion, support capacity, fault and geology, stress modelling, "
               "or event location uncertainty.")

# ------------------------------------------------------------------ tab 6: export
with tabs[5]:
    export = cat.copy()
    export["In_analysis"] = export["Class"].isin(include)
    st.download_button("Download classified catalogue (CSV)", csv_bytes(export),
                       "classified_catalogue.csv", "text/csv")
    st.download_button("Download daily rate table (CSV)", csv_bytes(daily),
                       "daily_rates.csv", "text/csv")
    st.dataframe(export.head(20))