# geoflux/pages/6_Conventional.py - GeoFlux Conventional Reservoirs (PROTOTYPE)
#
# Not validated. Resource estimates for screening, not reserves.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

import tabs_conv
import tabs_conv2

try:
    st.set_page_config(page_title="GeoFlux: Conventional (prototype)", layout="wide")
except Exception:
    pass

st.title("GeoFlux: Conventional Reservoirs")
st.warning("PROTOTYPE: not validated. Resource estimates for screening, not reserves. Use measured PVT and a "
           "qualified reservoir engineer before relying on any number here.")
tabs = st.tabs(["Volumetrics (OOIP / OGIP)", "Gas P/Z", "Oil material balance", "Notes & limits"])
with tabs[0]:
    tabs_conv.render_volumetrics()
with tabs[1]:
    tabs_conv2.render_pz()
with tabs[2]:
    tabs_conv2.render_oil_mb()
with tabs[3]:
    st.markdown(
        "**Methods**\n"
        "- Volumetrics: A x h x porosity x (1 - Sw) / formation volume factor, with triangular input ranges and a "
        "porosity-water saturation correlation.\n"
        "- Gas P/Z: straight line of p/z against cumulative gas; G is the intercept. Uncertainty from the regression.\n"
        "- Oil: Havlena-Odeh straight-line forms for depletion, a steady-state aquifer, and an unknown gas cap. "
        "Uncertainty from a Monte Carlo over measurement error.\n\n"
        "**Limits you should know**\n"
        "- P/Z ranges are well calibrated for straight-line data and overconfident when the data curves (the page flags it).\n"
        "- Aquifer fits cannot separate N from the aquifer constant well. Expect wide ranges.\n"
        "- Gas-cap fits run slightly narrow. The z-factor is an approximation. Average reservoir pressure must be "
        "measured, not wellhead pressure.\n"
        "- Tested only on synthetic reservoirs with known answers.")

# END OF FILE: 6_Conventional.py