# geoflux/pages/2_Decline_Models.py - GeoFlux Decline Models (PROTOTYPE)
#
# Not validated. Results only reflect the data and settings you supply.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import numpy as np
import streamlit as st

import geoflux_agg as agg
import geoflux_backtest as bt
import geoflux_dca as dca
import geoflux_demo as demo
import geoflux_io as gio
import geoflux_models as gm
import geoflux_runs as runs
import tabs_models

try:
    st.set_page_config(page_title="GeoFlux: Decline Models (prototype)", layout="wide")
except Exception:
    pass


@st.cache_data(show_spinner=False)
def demo_prod():
    return demo.make_synthetic_production(months=30)


@st.cache_data(show_spinner="Fitting models...")
def analyse(t, q, models, terminal):
    fits = {m: gm.fit(m, t, q) for m in models}
    fits = {m: f for m, f in fits.items() if f}
    origins = bt.default_origins(len(t))
    btdf = bt.backtest(t, q, list(fits), origins, terminal) if fits and origins else bt.backtest(t, q, [], [])
    return {"fits": fits, "cmp": bt.compare(fits) if fits else None, "origins": origins,
            "bt": btdf, "bts": bt.backtest_summary(btdf)}


@st.cache_data(show_spinner="Running Monte Carlo...")
def simulate(model, t, q, years, terminal, qlim, n, seed):
    return gm.probabilistic(model, gm.fit(model, t, q), n, years, terminal, qlim, seed)


st.sidebar.header("Data")
prod_file = st.sidebar.file_uploader("Production history (.csv)", type=["csv", "txt"],
                                     help="Date and Rate_boed (average BOE per day).")
if prod_file is not None:
    prod, err, notes = gio.load_production(prod_file)
    if err:
        st.error(f"Upload failed: {err}")
        st.stop()
    is_demo = False
else:
    prod, notes, is_demo = demo_prod(), [], True

st.sidebar.header("Models")
models = st.sidebar.multiselect("Models to fit", list(gm.MODELS), default=list(gm.MODELS), format_func=gm.label)
terminal = st.sidebar.slider("Arps terminal decline (effective %/year)", 0.0, 15.0, 6.0, 0.5)
years = st.sidebar.slider("Forecast horizon (years)", 5, 40, 20)
qlim = st.sidebar.number_input("Economic limit rate (BOE/d)", 0.0, 500.0, 10.0, 1.0)
n_samples = st.sidebar.slider("Monte Carlo samples", 100, 2000, 500, 100)
seed = int(st.sidebar.number_input("Random seed", 0, 10**6, 0, 1))
if not models:
    st.warning("Pick at least one model.")
    st.stop()

t = dca.production_time_axis(prod["Date"])
q = prod["Rate_boed"].to_numpy(dtype=float)
res = analyse(t, q, tuple(models), terminal)
if not res["fits"]:
    st.error("No model could be fitted. You need at least 6 to 8 valid production points.")
    st.stop()

fits, bts = res["fits"], res["bts"]
default_model = bts["Model"].iloc[0] if len(bts) else res["cmp"]["Model"].iloc[0]
keys = list(fits)
model_sel = st.sidebar.selectbox("Model for the forecast", keys, index=keys.index(default_model),
                                 format_func=gm.label, help="Defaults to the best hindcast, else the best AICc.")
sim = simulate(model_sel, t, q, years, terminal, qlim, n_samples, seed)

inputs = {"production": prod, "models": models, "terminal_pct": terminal, "horizon_years": years,
          "q_limit_boed": qlim, "n_samples": n_samples, "seed": seed, "forecast_model": model_sel}
outputs = {"fits": {m: {"params": f["params"], "aicc": f["aicc"], "r2": f["r2"]} for m, f in fits.items()},
           "backtest": bts, "eur_boe": agg.percentiles(sim["eur"])}
rec = runs.make_record("decline_models", inputs, outputs, seed)

st.title("GeoFlux: Decline Models")
st.warning("PROTOTYPE: not validated. Results only reflect the data and settings you supply. "
           "Nothing here is a reserves estimate.")
(st.warning if is_demo else st.info)(
    "PRODUCTION DATA SOURCE: " + ("SYNTHETIC DEMONSTRATION WELL (not real)" if is_demo
                                  else f"UPLOADED: {prod_file.name} ({len(prod)} points)"))
for n_ in notes:
    st.caption(f"Data note: {n_}")

S = SimpleNamespace(t=t, q=q, fits=fits, cmp=res["cmp"], bts=bts, bt=res["bt"], origins=res["origins"],
                    model_sel=model_sel, sim=sim, years=years, terminal=terminal, qlim=qlim, seed=seed, rec=rec)
tabs = st.tabs(["Fit & compare", "Backtest", "Forecast P90/P50/P10", "Diagnostics",
                "Portfolio effect", "Run record"])
with tabs[0]:
    tabs_models.render_compare(S)
with tabs[1]:
    tabs_models.render_backtest(S)
with tabs[2]:
    tabs_models.render_forecast(S)
with tabs[3]:
    tabs_models.render_diagnostics(S)
with tabs[4]:
    tabs_models.render_portfolio_explainer(S)
with tabs[5]:
    tabs_models.render_record(S)

# END OF FILE: 2_Decline_Models.py