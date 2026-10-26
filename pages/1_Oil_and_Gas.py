# geoflux/pages/1_Oil_and_Gas.py - GeoFlux Oil & Gas Module (PROTOTYPE)
#
# Not validated. Forecasts and economics only reflect the inputs you supply.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace

import streamlit as st

import geoflux_core as core
import geoflux_demo as demo
import geoflux_frac as frac
import geoflux_io as gio
import geoflux_mine as mine
import tabs_prod
import tabs_seis
from tabs_seis import csv_bytes

try:
    st.set_page_config(page_title="GeoFlux: Oil & Gas (prototype)", layout="wide")
except Exception:
    pass


@st.cache_data(show_spinner=False)
def demo_frac():
    cat, stages, well, fault, _truth = demo.make_synthetic_frac()
    return cat, stages, well, fault


@st.cache_data(show_spinner=False)
def demo_prod():
    return demo.make_synthetic_production(months=24)


# ----------------------------------------------------------------- sidebar: data
st.sidebar.header("Data")
cat_file = st.sidebar.file_uploader("Microseismic catalogue (.csv)", type=["csv", "txt"],
    help="Required: Time, X, Y, Z (same system as the wellbore) and Magnitude or Energy_J. "
         "Magnitudes are treated as moment magnitudes.")
stage_file = st.sidebar.file_uploader("Treatment stage log (.csv, optional)", type=["csv", "txt"],
    help="Stage, Start, End, and Volume_m3 or Volume_bbl. Optional: Max_Pressure_MPa or _psi.")
well_file = st.sidebar.file_uploader("Wellbore trajectory (.csv, optional)", type=["csv", "txt"],
    help="Name (optional), X, Y, Z points along the well.")
fault_file = st.sidebar.file_uploader("Faults (.csv, optional)", type=["csv", "txt"],
    help="Name, X, Y, Z points; rows with the same Name form a polyline.")
prod_file = st.sidebar.file_uploader("Production history (.csv, optional)", type=["csv", "txt"],
    help="Date and Rate_boed (average BOE per day).")

notes = []
if cat_file is not None:
    cat, err, w = mine.load_catalogue(cat_file)
    if err:
        st.error(f"Catalogue upload failed: {err}")
        st.stop()
    notes += w
    stages = wells = faults = None
    for f, name, loader in [(stage_file, "stages", gio.load_treatment_log),
                            (well_file, "wells", mine.load_workings),
                            (fault_file, "faults", mine.load_workings)]:
        if f is not None:
            obj, err, w = loader(f)
            if err:
                st.error(f"Upload failed ({f.name}): {err}")
                st.stop()
            notes += w
            if name == "stages":
                stages = obj
            elif name == "wells":
                wells = obj
            else:
                faults = obj
    frac_demo = False
    source = f"UPLOADED: {cat_file.name} ({len(cat):,} events)"
else:
    cat, stages, wells, faults = demo_frac()
    frac_demo = True
    source = "SYNTHETIC DEMONSTRATION FRAC PAD (randomly generated; not a real well)"
    if any(f is not None for f in (stage_file, well_file, fault_file)):
        notes.append("Stage log / wellbore / faults are ignored until a catalogue is uploaded.")

if prod_file is not None:
    prod, err, w = gio.load_production(prod_file)
    if err:
        st.error(f"Production upload failed: {err}")
        st.stop()
    notes += w
    prod_demo = False
else:
    prod, prod_demo = demo_prod(), True

with st.sidebar.expander("Download CSV templates"):
    d_cat, d_st, d_well, d_fault = demo_frac()
    for label, data, fn in [("Catalogue", d_cat.head(200), "frac_catalogue_template.csv"),
                            ("Stage log", d_st, "stage_log_template.csv"),
                            ("Wellbore", d_well, "wellbore_template.csv"),
                            ("Faults", d_fault, "faults_template.csv"),
                            ("Production", demo_prod(), "production_template.csv")]:
        st.download_button(f"{label} template", csv_bytes(data), fn, "text/csv")

# --------------------------------------------------------- sidebar: seismicity
st.sidebar.header("Seismicity settings")
lag_min = st.sidebar.number_input("Post-pumping lag assigned to a stage (min)", 0, 480, 60)
G_gpa = st.sidebar.number_input("Shear modulus G (GPa)", 5.0, 60.0, 30.0, 1.0,
                                help="Used for the McGarr moment bound. About 30 GPa is typical.")
amber = st.sidebar.number_input("Traffic light: amber at magnitude", value=1.0, step=0.1)
red = st.sidebar.number_input("Traffic light: red at magnitude", value=2.0, step=0.1)
st.sidebar.caption("Placeholder thresholds. Set them from your permit or regulator.")
if amber >= red:
    st.sidebar.error("Amber must be below red.")
    st.stop()
min_vol = float(stages["Volume_m3"].median()) if stages is not None else 600.0
next_vol = st.sidebar.number_input("Next-stage volume (m3) for look-ahead", 10.0, 20000.0, min_vol, 10.0)

st.sidebar.header("SRV and geometry")
voxel = st.sidebar.slider("Voxel size (m)", 5, 100, 20, 5)
min_ev = st.sidebar.slider("Minimum events per voxel", 1, 10, 2)
fault_thr = st.sidebar.slider("Near-fault distance (m)", 10, 500, 100, 10)

# ------------------------------------------------------------ sidebar: production
st.sidebar.header("Decline and economics")
skip_n = st.sidebar.slider("Ignore first N production points", 0, 6, 0)
b_lo, b_hi = st.sidebar.slider("Allowed Arps b range", 0.05, 2.0, (0.05, 2.0), 0.05)
dmin_pct = st.sidebar.slider("Terminal decline (effective %/year)", 0.0, 15.0, 6.0, 0.5)
horizon_y = st.sidebar.slider("Forecast horizon (years)", 5, 40, 20)
n_mc = st.sidebar.slider("Monte Carlo samples", 100, 1000, 300, 100)
price = st.sidebar.number_input("Realised price per BOE", 0.0, 500.0, 60.0, 1.0)
royalty = st.sidebar.number_input("Royalty (%)", 0.0, 50.0, 12.5, 0.5)
var_opex = st.sidebar.number_input("Variable opex per BOE", 0.0, 200.0, 8.0, 0.5)
fixed_opex = st.sidebar.number_input("Fixed opex per month", 0.0, 1e7, 15000.0, 1000.0)
capex_m = st.sidebar.number_input("Capex (millions)", 0.0, 500.0, 10.0, 0.5)
disc = st.sidebar.number_input("Discount rate (%/year)", 0.0, 50.0, 10.0, 0.5)
tax = st.sidebar.number_input("Tax on positive operating cash flow (%)", 0.0, 60.0, 0.0, 1.0)
st.sidebar.caption("Use one consistent currency for price, opex and capex.")
econ = dict(royalty_pct=royalty, var_opex=var_opex, fixed_opex_month=fixed_opex,
            capex=capex_m * 1e6, disc_rate_pct=disc, tax_pct=tax)

# ---------------------------------------------------------------- computations
cat = cat.copy()
have_stages = stages is not None
if have_stages:
    cat["Stage"] = frac.assign_stages(cat["Time"], stages, lag_min)
mag_bin = core.detect_mag_bin(cat["Magnitude"])
mc = core.estimate_mc_maxc(cat["Magnitude"], mag_bin)
b_res = core.estimate_b_value(cat["Magnitude"], mc, mag_bin) if mc is not None else None
seis_ok = b_res is not None
summ = (frac.stage_summary(cat, stages, mc, b_res["b"], G_gpa * 1e9, amber, red)
        if seis_ok and have_stages else None)
if faults is not None:
    cat["Dist_to_fault_m"], cat["Nearest_fault"] = mine.distance_to_workings(cat, faults)

# --------------------------------------------------------------------- header
st.title("GeoFlux: Oil & Gas Module")
st.warning("PROTOTYPE: not validated. Seismicity screening does not replace your regulator-approved "
           "monitoring plan. Forecasts and economics only reflect the inputs you supply: no reservoir "
           "properties are modelled.")
(st.warning if frac_demo else st.info)(f"MICROSEISMIC DATA SOURCE: {source}")
(st.warning if prod_demo else st.info)(
    "PRODUCTION DATA SOURCE: " + ("SYNTHETIC DEMONSTRATION WELL (not real)" if prod_demo
                                  else f"UPLOADED: {prod_file.name} ({len(prod)} points)"))
for n_ in notes:
    st.caption(f"Data note: {n_}")

tabs = st.tabs(["Data & QC", "Stages & seismicity", "Traffic light", "SRV & geometry",
                "Production & economics", "Export"])


S = SimpleNamespace(
    cat=cat, stages=stages, wells=wells, faults=faults, prod=prod, have_stages=have_stages,
    mc=mc, b_res=b_res, seis_ok=seis_ok, summ=summ, lag_min=lag_min, G_gpa=G_gpa,
    amber=amber, red=red, next_vol=next_vol, voxel=voxel, min_ev=min_ev, fault_thr=fault_thr,
    skip_n=skip_n, b_lo=b_lo, b_hi=b_hi, dmin_pct=dmin_pct, horizon_y=horizon_y, n_mc=n_mc,
    price=price, econ=econ)

with tabs[0]:
    tabs_seis.render_qc(S)
with tabs[1]:
    tabs_seis.render_stages(S)
with tabs[2]:
    tabs_seis.render_traffic(S)
with tabs[3]:
    tabs_prod.render_srv(S)
with tabs[4]:
    prod_out = tabs_prod.render_production(S)
with tabs[5]:
    tabs_prod.render_export(S, prod_out)

# END OF FILE: 1_Oil_and_Gas.py