# geoflux/pages/3_Economics_Risk.py - GeoFlux Economics & Risk (PROTOTYPE)
#
# Not validated. A screening estimate that only reflects the inputs you supply.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import numpy as np
import streamlit as st

import geoflux_agg as agg
import geoflux_dca as dca
import geoflux_demo as demo
import geoflux_econ as econ
import geoflux_io as gio
import geoflux_models as gm
import geoflux_risk as risk
import geoflux_runs as runs
import tabs_econ

try:
    st.set_page_config(page_title="GeoFlux: Economics & Risk (prototype)", layout="wide")
except Exception:
    pass


@st.cache_data(show_spinner=False)
def demo_prod():
    return demo.make_synthetic_production(months=30)


@st.cache_data(show_spinner="Simulating decline uncertainty...")
def simulate(model, t, q, years, terminal, qlim, n, seed):
    f = gm.fit(model, t, q)
    return (f, gm.probabilistic(model, f, n, years, terminal, qlim, seed)) if f else (None, None)


st.sidebar.header("Production data")
prod_file = st.sidebar.file_uploader("Production history (.csv)", type=["csv", "txt"], help="Date and Rate_boed (BOE/d).")
if prod_file is not None:
    prod, err, notes = gio.load_production(prod_file)
    if err:
        st.error(f"Upload failed: {err}")
        st.stop()
    is_demo = False
else:
    prod, notes, is_demo = demo_prod(), [], True

st.sidebar.header("Forecast")
model = st.sidebar.selectbox("Decline model", list(gm.MODELS), format_func=gm.label)
terminal = st.sidebar.slider("Arps terminal decline (effective %/year)", 0.0, 15.0, 6.0, 0.5)
years = st.sidebar.slider("Horizon (years)", 5, 40, 20)
qlim = st.sidebar.number_input("Economic limit rate (BOE/d)", 0.0, 500.0, 10.0, 1.0)
n_samples = st.sidebar.slider("Monte Carlo samples", 100, 2000, 500, 100)
seed = int(st.sidebar.number_input("Random seed", 0, 10**6, 0, 1))

st.sidebar.header("Products and prices")
gas_share = st.sidebar.slider("Gas share of BOE (%)", 0, 100, 30) / 100.0
oil_price = st.sidebar.number_input("Oil price per bbl", 0.0, 500.0, 65.0, 1.0)
gas_price = st.sidebar.number_input("Gas price per Mcf", 0.0, 50.0, 3.0, 0.1)
oil_diff = st.sidebar.number_input("Oil differential per bbl", -50.0, 50.0, -3.0, 0.5)
gas_diff = st.sidebar.number_input("Gas differential per Mcf", -10.0, 10.0, -0.3, 0.1)
ngl_yield = st.sidebar.number_input("NGL yield (bbl per MMcf)", 0.0, 200.0, 0.0, 1.0)
ngl_pct = st.sidebar.number_input("NGL price (% of oil)", 0.0, 100.0, 35.0, 1.0)
shrink = st.sidebar.number_input("Gas shrinkage (%)", 0.0, 50.0, 0.0, 1.0) / 100.0

st.sidebar.header("Costs and fiscal terms")
royalty = st.sidebar.number_input("Royalty (%)", 0.0, 50.0, 12.5, 0.5)
ptax = st.sidebar.number_input("Production tax (% of net revenue)", 0.0, 30.0, 5.0, 0.5)
itax = st.sidebar.number_input("Income tax (%)", 0.0, 60.0, 27.0, 1.0, help="Placeholder. Set from your fiscal terms.")
dep_years = st.sidebar.number_input("Depreciation period (years)", 0.0, 30.0, 7.0, 1.0)
var_opex = st.sidebar.number_input("Variable opex per BOE", 0.0, 200.0, 8.0, 0.5)
fixed_opex = st.sidebar.number_input("Fixed opex per month", 0.0, 1e7, 15000.0, 1000.0)
capex_m = st.sidebar.number_input("Capex (millions)", 0.0, 500.0, 10.0, 0.5)
abandon = st.sidebar.number_input("Abandonment cost", 0.0, 1e8, 150000.0, 10000.0)
disc = st.sidebar.number_input("Discount rate (%/year)", 0.0, 50.0, 10.0, 0.5)

st.sidebar.header("Uncertainty ranges (% of base)")
rng_in = {k: st.sidebar.slider(lab, 20, 250, lohi, 5) for k, lab, lohi in [
    ("oil_price", "Oil price", (70, 130)), ("gas_price", "Gas price", (60, 140)), ("capex", "Capex", (85, 115)),
    ("var_opex", "Variable opex", (80, 120)), ("fixed_opex_month", "Fixed opex", (80, 120))]}
price_corr = st.sidebar.slider("Oil-gas price correlation", 0.0, 1.0, 0.5, 0.1)

deck = {"oil_price": oil_price, "gas_price": gas_price, "oil_diff": oil_diff, "gas_diff": gas_diff,
        "gas_share": gas_share, "ngl_yield": ngl_yield, "ngl_price_pct": ngl_pct, "gas_shrink": shrink,
        "royalty_pct": royalty, "prod_tax_pct": ptax, "var_opex": var_opex, "fixed_opex_month": fixed_opex,
        "capex": capex_m * 1e6, "abandon_cost": abandon, "tax_pct": itax, "dep_years": dep_years, "disc_rate_pct": disc}
spec = {k: (deck[k] * lo / 100.0, deck[k], deck[k] * hi / 100.0) for k, (lo, hi) in rng_in.items()}

t = dca.production_time_axis(prod["Date"])
q = prod["Rate_boed"].to_numpy(dtype=float)
fit, sim = simulate(model, t, q, years, terminal, qlim, n_samples, seed)
if fit is None:
    st.error("The decline model could not be fitted. Try another model or more production points.")
    st.stop()

base_rates = gm.forecast_monthly(model, fit["params"], years, terminal)
mc = risk.monte_carlo(sim["rates"], deck, spec, seed, price_corr)
p90, p10 = risk.pick_case(sim["rates"], sim["eur"], 10), risk.pick_case(sim["rates"], sim["eur"], 90)
tor_base, tor_df = risk.tornado(base_rates, p90, p10, deck, spec)
cases = {"EUR P90 case": p90, "Best fit": base_rates, "EUR P10 case": p10}
curve_df = risk.price_curve(cases, deck, np.linspace(0.3, 2.0, 18))
be_cases = {}
for name, qq in cases.items():
    m = econ.breakeven_multiplier(qq, deck)
    be_cases[name] = m * oil_price if m is not None else None
eco = econ.cash_flow(base_rates, deck)

inputs = {"production": prod, "model": model, "terminal_pct": terminal, "horizon_years": years,
          "q_limit": qlim, "n_samples": n_samples, "seed": seed, "deck": deck, "ranges": spec, "price_corr": price_corr}
outputs = {"npv": risk.summarise(mc["npv"]), "eur_boe": agg.percentiles(sim["eur"]), "base_npv": eco["npv"],
           "breakeven_oil_price": be_cases}
rec = runs.make_record("economics_risk", inputs, outputs, seed)

st.title("GeoFlux: Economics & Risk")
st.warning("PROTOTYPE: not validated. A screening estimate that only reflects the inputs you supply. "
           "Not a valuation, a reserves report, or tax advice.")
(st.warning if is_demo else st.info)("PRODUCTION DATA SOURCE: " + (
    "SYNTHETIC DEMONSTRATION WELL (not real)" if is_demo else f"UPLOADED: {prod_file.name} ({len(prod)} points)"))
for n_ in notes:
    st.caption(f"Data note: {n_}")

S = SimpleNamespace(eco=eco, summ=risk.summarise(mc["npv"]), eur_p=agg.percentiles(sim["eur"]), mc=mc, deck=deck,
                    spec=spec, tor_base=tor_base, tor_df=tor_df, curve_df=curve_df, be_cases=be_cases,
                    be_oil=be_cases["Best fit"], rec=rec)
tabs = st.tabs(["Summary", "Tornado", "Price sensitivity", "Cash flow", "Assumptions & record"])
with tabs[0]:
    tabs_econ.render_summary(S)
with tabs[1]:
    tabs_econ.render_tornado(S)
with tabs[2]:
    tabs_econ.render_price(S)
with tabs[3]:
    tabs_econ.render_cashflow(S)
with tabs[4]:
    tabs_econ.render_record(S)

# END OF FILE: 3_Economics_Risk.py