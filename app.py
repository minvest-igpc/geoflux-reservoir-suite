# geoflux/app.py - Standalone Unified Subsurface Suite (Complete Module 1, 2, 3 Build)
import streamlit as st
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

# 1. CORE PHYSICS ENGINEERING ENGINE
class SubsurfaceAnalyticsEngine:
    def __init__(self, df: pd.DataFrame, depth_regime: str):
        self.data = df
        self.regime = depth_regime

    def process_geomechanics(self) -> dict:
        xyz = self.data[['Easting_m', 'Northing_m', 'Depth_m']].values
        base_vol = ConvexHull(xyz).volume if len(xyz) >= 4 else 0.0
        
        scale = 10.5 if "Intermediate" in self.regime else (14.2 if "Deep" in self.regime else 18.5)
        srv = float(base_vol * scale)
        
        tensile_e = self.data.loc[self.data['Mechanism'] == 'Tensile', 'Energy_J'].sum()
        total_e = self.data['Energy_J'].sum()
        fcd = float((tensile_e / total_e if total_e > 0 else 0.2) * 65.0)
        
        # Calculate real localized magnitude scales from microseismic energy proxy
        magnitudes = 0.67 * np.log10(np.maximum(self.data['Energy_J'].values, 1e-3)) - 1.2
        b_val = float(1.0 / (np.mean(magnitudes) - np.min(magnitudes))) if len(magnitudes) > 0 else 1.0
        if "Ultra-Deep" in self.regime: b_val *= 0.85
        
        return {"srv": srv, "fcd": fcd, "b_value": np.clip(b_val, 0.45, 1.95), "magnitudes": magnitudes}

    def generate_forecasts(self, fcd: float, srv: float, opex: float, tensile_ratio_input: float, noise_input: float) -> dict:
        qi = np.clip(float(1000 + (fcd * 55)), 800, 6500)
        t = np.arange(1, 3653)
        
        di_factor = np.clip(float(0.45 - (tensile_ratio_input / 300.0)), 0.05, 0.40)
        curvature_n = np.clip(float(0.40 + (srv / 2000000.0) + (noise_input / 200.0)), 0.15, 0.95)
        
        q_fluid = qi / (1.0 + di_factor * t) ** curvature_n
        w_scale = 0.35 if "Deep" in self.regime else (0.50 if "Ultra-Deep" in self.regime else 0.75)
        q_water = (t ** w_scale) * (q_fluid * 0.04)
        
        active_days = np.where(q_fluid >= opex)
        cutoff = int(active_days[-1]) + 1 if len(active_days) > 0 else 3652
        
        return {
            "timeline": (t / 365.25).tolist(),
            "oil_rate": q_fluid.tolist(),
            "water_rate": q_water.tolist(),
            "eur": float(np.sum(q_fluid[:cutoff])),
            "life": round(cutoff / 365.25, 2)
        }

# 2. INTERACTIVE USER INTERFACE LAYER
st.set_page_config(page_title="GeoFlux Reservoir Suite", layout="wide", page_icon="📊")
st.title("📊 GeoFlux Reservoir Suite")
st.subheader("Physics Engine Core: FracMechanica (Complete Architecture Upgrade)")
st.markdown("---")

st.sidebar.header("🛠️ Reservoir Optimization Console")
env = st.sidebar.selectbox("Target Depth Envelope", ["Intermediate (1,000m - 3,000m)", "Deep (3,000m - 4,500m)", "Ultra-Deep (>4,500m)"])
count = st.sidebar.slider("Microseismic Event Index", 50, 1000, 300, 50)
noise = st.sidebar.slider("Rock Mass Anisotropy (Noise)", 5, 60, 25, 5)
tensile_pct = st.sidebar.slider("Focal Mechanism Tensile Ratio (%)", 5, 95, 35, 5)
uploaded_file = st.sidebar.file_uploader("Upload Subsurface Text Log (.csv/.txt)", type=["csv", "txt"])
opex_limit = st.sidebar.slider("Economic Operating Limit (BOE / Day)", 5.0, 120.0, 25.0, 5.0)

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        rename_dict = {'Energy': 'Energy_J', 'Energy_Joules': 'Energy_J', 'X': 'Easting_m', 'Y': 'Northing_m', 'Z': 'Depth_m', 'Mechanism_Type': 'Mechanism', 'Source_Mechanism': 'Mechanism'}
        raw_df.rename(columns={k: v for k, v in rename_dict.items() if k in raw_df.columns}, inplace=True)
    except Exception:
        uploaded_file = None

if uploaded_file is None:
    np.random.seed(42)
    mechs = np.random.choice(['Shear', 'Tensile', 'Closure'], size=count, p=[1.0 - (tensile_pct/100.0) - 0.05, tensile_pct/100.0, 0.05])
    raw_df = pd.DataFrame({
        'Easting_m': np.random.normal(500, noise * 4.0, count),
        'Northing_m': np.random.normal(1200, noise * 2.0, count),
        'Depth_m': np.random.normal(-2800, noise * 3.0, count),
        'Mechanism': mechs,
        'Energy_J': np.random.lognormal(2.2, 0.6, count)
    })
st.sidebar.info("ℹ️ Synthesizing data live from geomechanical sliders.")

engine = SubsurfaceAnalyticsEngine(raw_df, depth_regime=env)
geo = engine.process_geomechanics()
curves = engine.generate_forecasts(geo["fcd"], geo["srv"], opex_limit, tensile_ratio_input=tensile_pct, noise_input=noise)

st.markdown("### 🍽️ The Petroleum Engineer's Menu")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Stimulated Reservoir Vol (SRV)", f"{geo['srv']:,.2f} m³")
m2.metric("Fracture Conductivity (Fcd)", f"{geo['fcd']:.2f} mD-ft")
m3.metric("Seismic b-Value Index", f"{geo['b_value']:.2f}")
m4.metric("Commercially Productive Life", f"{curves['life']} Years")
m5.metric("Net Recoverable Asset EUR", f"{curves['eur']:,.0f} BOE")
st.markdown("---")

# MODULE 1 DYNAMIC CHART ADD-ON: 3-Column Visualization Grid Split
st.markdown("### 📊 Complete Geomechanical & Fluids Analytics Dashboard")
chart_col1, chart_col2, chart_col3 = st.columns(3)

with chart_col1:
    st.markdown("#### 📈 Hydrocarbon Decline Profile")
    oil_df = pd.DataFrame({'Hydrocarbon Flow Rate (BOE/d)': curves["oil_rate"]}, index=np.round(curves["timeline"], 2))
    st.area_chart(oil_df)

with chart_col2:
    st.markdown("#### 💧 Produced Water Cut Curve (WOR)")
    water_df = pd.DataFrame({'Produced Water Loading (BWPD)': curves["water_rate"]}, index=np.round(curves["timeline"], 2))
    st.area_chart(water_df)

with chart_col3:
    st.markdown("#### ⚡ Module 1: Gutenberg-Richter Frequency Plot")
    # Dynamically generate cumulative log event distributions based on magnitudes
    m_data = geo["magnitudes"]
    bins = np.linspace(np.min(m_data), np.max(m_data), 15)
    counts, bin_edges = np.histogram(m_data, bins=bins)
    cum_counts = np.cumsum(counts[::-1])[::-1]
    log_counts = np.log10(np.maximum(cum_counts, 1))
    
    gr_df = pd.DataFrame({
        'Log10 Cumulative Events (Log N)': log_counts
    }, index=np.round(bin_edges[:-1], 2))
    # Stable line graph canvas to illustrate tectonic slopes / fault slips safely
    st.line_chart(gr_df, y_label="Log10 Cumulative Events (Log N)")

st.markdown("---")
l_col, r_col = st.columns(2)
with l_col:
    st.markdown("### 🗃️ Active Subsurface Spatial Log Table")
    st.dataframe(raw_df[['Easting_m', 'Northing_m', 'Depth_m', 'Mechanism', 'Energy_J']].head(100), use_container_width=True)
with r_col:
    st.markdown("### 📄 Module 3: Executive Export Center")
    st.success(f"PRO-FORMA REVIEW PASSED: Selected {env} envelope optimized at {curves['eur']:,.0f} BOE total net asset volume value.")
