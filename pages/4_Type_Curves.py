# geoflux/pages/4_Type_Curves.py - GeoFlux Type Curves from analog wells (PROTOTYPE)
#
# Not validated. Results only reflect the wells and settings you supply.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import numpy as np
import streamlit as st

import geoflux_agg as agg
import geoflux_models as gm
import geoflux_runs as runs
import geoflux_typecurve as tc
import geoflux_wells as wl
import tabs_tc
import tabs_tc2

try:
    st.set_page_config(page_title="GeoFlux: Type Curves (prototype)", layout="wide")
except Exception:
    pass

prod, meta, is_demo, notes, source = tabs_tc.load_wells_sidebar()
series_all = wl.well_series(prod)

st.sidebar.header("Analog filters")
forms = sorted(meta["Formation"].dropna().unique()) if "Formation" in meta else []
sel_forms = st.sidebar.multiselect("Formation", forms, default=forms) if forms else None
years_rng = lat_rng = types = None
if "Completion_Year" in meta and meta["Completion_Year"].notna().any():
    y0, y1 = int(meta["Completion_Year"].min()), int(meta["Completion_Year"].max())
    years_rng = st.sidebar.slider("Completion year", y0, y1, (y0, y1)) if y1 > y0 else (y0, y1)
if "Lateral_m" in meta and meta["Lateral_m"].notna().any():
    l0, l1 = int(meta["Lateral_m"].min()), int(meta["Lateral_m"].max()) + 1
    lat_rng = st.sidebar.slider("Lateral length (m)", l0, l1, (l0, l1)) if l1 > l0 else (l0, l1)
if "Well_Type" in meta:
    wt = sorted(meta["Well_Type"].dropna().unique())
    types = st.sidebar.multiselect("Well type", wt, default=wt)

st.sidebar.header("Model and settings")
model = st.sidebar.selectbox("Decline model", list(gm.MODELS), format_func=gm.label)
terminal = st.sidebar.slider("Arps terminal decline (effective %/year)", 0.0, 15.0, 6.0, 0.5)
years = st.sidebar.slider("Horizon (years)", 5, 40, 30)
qlim = st.sidebar.number_input("Economic limit rate (BOE/d)", 0.0, 500.0, 10.0, 1.0)
min_points = st.sidebar.slider("Minimum months of data per well", 6, 36, 12)
min_r2 = st.sidebar.slider("Minimum fit R2", 0.5, 0.99, 0.80, 0.01)
min_wells = st.sidebar.slider("Minimum wells per month (percentile curves)", 3, 20, 5)
beta = st.sidebar.slider("Lateral-length exponent", 0.3, 1.3, 1.0, 0.05, help="1.0: volume proportional to length.")
n_samples = st.sidebar.slider("Samples", 200, 5000, 1000, 200)
seed = int(st.sidebar.number_input("Random seed", 0, 10**6, 0, 1))

names = [w for w in wl.select_analogs(meta, sel_forms, years_rng, lat_rng, types) if w in series_all]
if not names:
    st.error("No wells match the filters.")
    st.stop()
series = {w: series_all[w] for w in names}
lateral = None
if "Lateral_m" in meta and meta.set_index("Well").loc[names, "Lateral_m"].notna().all():
    lateral = {w: float(v) for w, v in meta.set_index("Well").loc[names, "Lateral_m"].items()}
fits, rejected = tabs_tc.fit_all(series, model, terminal, years, qlim, min_points, min_r2)
if not fits:
    st.error("No well could be fitted with these settings. Lower the minimum months or R2.")
    st.stop()

lat_default = float(np.median(list(lateral.values()))) if lateral else 1000.0
lateral_new = st.sidebar.number_input("New well lateral length (m)", 100.0, 10000.0, lat_default, 50.0)
pc = tc.pcurves(series, list(fits), lateral, min_wells)
if pc.empty:
    st.error("Not enough wells per month for percentile curves. Lower the minimum wells per month.")
    st.stop()
tc_fits = {c: tc.fit_type_curve(pc, model, c) for c in ["P90", "P50", "P10", "Mean"]}
samples = tc.analog_samples(fits, lateral, n_samples, lateral_new if lateral else None, beta, seed) if len(fits) >= 5 else None
coverage = tc.loo_coverage(fits, lateral, beta)

inputs = {"production": prod[prod["Well"].isin(names)], "metadata": meta, "model": model, "terminal_pct": terminal,
          "horizon_years": years, "q_limit": qlim, "min_points": min_points, "min_r2": min_r2, "beta": beta,
          "lateral_new_m": lateral_new, "n_samples": n_samples, "seed": seed}
outputs = {"analogs_fitted": list(fits), "rejected": rejected, "loo_coverage": coverage[0],
           "eur_boe": agg.percentiles(samples["eur"]) if samples else None}
rec = runs.make_record("type_curves", inputs, outputs, seed)

st.title("GeoFlux: Type Curves from Analog Wells")
st.warning("PROTOTYPE: not validated. Results only reflect the wells and settings you supply. Not a reserves estimate.")
(st.warning if is_demo else st.info)(f"WELL DATA SOURCE: {source}")
for n_ in notes:
    st.caption(f"Data note: {n_}")

S = SimpleNamespace(n_total=len(series_all), n_selected=len(names), fits=fits, rejected=rejected, pc=pc, model=model,
                    terminal=terminal, years=years, qlim=qlim, normalised=lateral is not None, tc_fits=tc_fits,
                    well_tab=tc.well_table(fits, meta), samples=samples, lateral_new=lateral_new, beta=beta, seed=seed,
                    coverage=coverage, rec=rec)
tabs = st.tabs(["Analog wells", "Percentile curves", "New well forecast", "Calibration & limits", "Run record"])
with tabs[0]:
    tabs_tc.render_analogs(S)
with tabs[1]:
    tabs_tc.render_pcurves(S)
with tabs[2]:
    tabs_tc2.render_new_well(S)
with tabs[3]:
    tabs_tc2.render_quality(S)
with tabs[4]:
    tabs_tc2.render_record(S)

# END OF FILE: 4_Type_Curves.py