# geoflux/pages/5_Portfolio.py - GeoFlux Portfolio of planned wells (PROTOTYPE)
#
# Not validated. Results only reflect the analog wells and assumptions you supply.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

import geoflux_agg as agg
import geoflux_models as gm
import geoflux_port as gp
import geoflux_runs as runs
import geoflux_wells as wl
import tabs_port
import tabs_tc

try:
    st.set_page_config(page_title="GeoFlux: Portfolio (prototype)", layout="wide")
except Exception:
    pass

prod, meta, is_demo, notes, source = tabs_tc.load_wells_sidebar()
series = wl.well_series(prod)

st.sidebar.header("Analog model")
model = st.sidebar.selectbox("Decline model", list(gm.MODELS), format_func=gm.label)
terminal = st.sidebar.slider("Arps terminal decline (effective %/year)", 0.0, 15.0, 6.0, 0.5)
years = st.sidebar.slider("Horizon (years)", 5, 40, 30)
qlim = st.sidebar.number_input("Economic limit rate (BOE/d)", 0.0, 500.0, 10.0, 1.0)
min_points = st.sidebar.slider("Minimum months of data per well", 6, 36, 12)
min_r2 = st.sidebar.slider("Minimum fit R2", 0.5, 0.99, 0.80, 0.01)
beta = st.sidebar.slider("Lateral-length exponent", 0.3, 1.3, 1.0, 0.05, help="1.0: volume proportional to length.")
st.sidebar.header("Portfolio risk")
rho = st.sidebar.slider("Correlation between wells", 0.0, 1.0, 0.3, 0.05,
                        help="A visible assumption. 0 means independent wells, which overstates diversification.")
n_samples = st.sidebar.slider("Samples", 500, 5000, 2000, 500)
seed = int(st.sidebar.number_input("Random seed", 0, 10**6, 0, 1))

fits, rejected = tabs_tc.fit_all(series, model, terminal, years, qlim, min_points, min_r2)
if not fits:
    st.error("No well could be fitted with these settings.")
    st.stop()

forms = sorted(meta["Formation"].dropna().unique()) if "Formation" in meta else []
lat_med = float(meta["Lateral_m"].median()) if "Lateral_m" in meta and meta["Lateral_m"].notna().any() else 2000.0
if is_demo:
    default_plan = pd.DataFrame([{"Group": "Block A", "Formation": "Formation A", "Lateral_m": 2500.0, "Wells": 6},
                                 {"Group": "Block B", "Formation": "Formation B", "Lateral_m": 3000.0, "Wells": 4}])
else:
    default_plan = pd.DataFrame([{"Group": "Plan 1", "Formation": "All", "Lateral_m": lat_med, "Wells": 5}])

st.title("GeoFlux: Portfolio of Planned Wells")
st.warning("PROTOTYPE: not validated. Results only reflect the analog wells and assumptions you supply. Not a reserves estimate.")
(st.warning if is_demo else st.info)(f"WELL DATA SOURCE: {source}")
for n_ in notes:
    st.caption(f"Data note: {n_}")

st.subheader("Development plan (edit the table)")
edited = st.data_editor(default_plan, num_rows="dynamic", hide_index=True, key="plan", column_config={
    "Group": st.column_config.TextColumn("Group"),
    "Formation": st.column_config.SelectboxColumn("Formation", options=["All"] + forms, required=True),
    "Lateral_m": st.column_config.NumberColumn("Lateral (m)", min_value=100.0, max_value=10000.0, step=50.0),
    "Wells": st.column_config.NumberColumn("Wells", min_value=1, max_value=200, step=1)})
plan = edited.dropna()
plan = plan[(plan["Wells"] >= 1) & (plan["Lateral_m"] > 0) & (plan["Group"].astype(str).str.strip() != "")].copy()
plan["Wells"] = plan["Wells"].astype(int)
if plan.empty:
    st.info("Add at least one row to the plan.")
    st.stop()
if plan["Wells"].sum() > 300:
    st.error("Too many wells (limit 300).")
    st.stop()

names, groups, X, problems = gp.build_samples(plan.to_dict("records"), fits, meta, n_samples, beta, seed)
for p_ in problems:
    st.warning(p_)
if X.shape[1] == 0:
    st.error("No planned well could be sampled.")
    st.stop()

groups_tbl, Y = gp.group_tables(X, groups, names, rho, seed)
total = Y.sum(axis=1)
well_tbl = agg.well_table({nm: X[:, k] for k, nm in enumerate(names)})
well_tbl.insert(1, "Group", groups)
arith, prob = agg.arithmetic_sum(well_tbl), agg.percentiles(total)
diff = {k: (100.0 * (prob[k] / arith[k] - 1.0) if arith[k] else float("nan")) for k in ["P90", "P50", "P10"]}
curve = gp.correlation_curve(X, seed=seed)

rec = runs.make_record("portfolio", {"plan": plan, "model": model, "terminal_pct": terminal, "horizon_years": years,
                                     "q_limit": qlim, "min_points": min_points, "min_r2": min_r2, "beta": beta,
                                     "rank_correlation": rho, "n_samples": n_samples, "seed": seed,
                                     "analogs": sorted(fits)},
                       {"arithmetic": arith, "probabilistic": prob, "difference_pct": diff}, seed)
S = SimpleNamespace(arith=arith, prob=prob, diff=diff, total=total, groups=groups_tbl, curve=curve, plan=plan,
                    well_tbl=well_tbl, n_wells=len(names), rho=rho, rec=rec)
tabs = st.tabs(["Portfolio EUR", "Correlation sensitivity", "Wells & plan", "Run record"])
with tabs[0]:
    tabs_port.render_total(S)
with tabs[1]:
    tabs_port.render_corr(S)
with tabs[2]:
    tabs_port.render_wells(S)
with tabs[3]:
    tabs_port.render_record(S)

# END OF FILE: 5_Portfolio.py