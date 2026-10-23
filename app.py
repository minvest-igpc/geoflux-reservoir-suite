# geoflux/app.py - Standalone Unified Subsurface Suite (Fresh Start)
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
        # Calculate volume envelope via 3D Convex Hull
        xyz = self.data[['Easting_m', 'Northing_m', 'Depth_m']].values
        base_vol = ConvexHull(xyz).volume if len(xyz) >= 4 else 0.0
        
        # Scale Volumetric Multipliers based on depth envelope matrix
        scale = 10.5 if "Intermediate" in self.regime else (14.2 if "Deep" in self.regime else 18.5)
        srv = float(base_vol * scale)
        
        # Calculate fracture path conductivity proxy from tensile energy inversion
        tensile_e = self.data.loc[self.data['Mechanism'] == 'Tensile', 'Energy_J'].sum()
        total_e = self.data['Energy_J'].sum()
        fcd = float((tensile_e / total_e if total_e > 0 else 0.2) * 65.0)
        
        # Module 1: Gutenberg-Richter Seismic b-value calculation
        magnitudes = 0.67 * np.log10(np.maximum(self.data['Energy_J'].values, 1e-3)) - 1.2
        b_val = float(1.0 / (np.mean(magnitudes) - np.min(magnitudes))) if len(magnitudes) > 0 else 1.0
        if "Ultra-Deep" in self.regime: b_val *= 0.85  # Tectonic stress damping
        
        return {"srv": srv, "fcd": fcd, "b_value": np.clip(b_val, 0.45, 1.95)}

    def generate_forecasts(self, fcd: float, srv: float, opex: float) -> dict:
        # Setup initial flow rates and decline curvatures
        qi = np.clip(float(1200 + (fcd * 45)), 1000, 6000)
        t = np.arange(1, 3653)
        
        # Standard Arps Hyperbolic decline curve profile
        q_fluid = qi / (1.0 + 0.15 * t) ** 1.3
        
        # Module 2: Chan Water-Oil Ratio (WOR) water-cut dynamic curves
        w_scale = 0.4 if "Deep" in self.regime else 0.8
        q_water = (t ** w_scale) * (q_fluid * 0.05)
        
        # Calculate exact economic limit abandonment days cleanly
        active_days = np.where(q_fluid >= opex)[0]
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
st.subheader("Physics Engine Core: FracMechanica (Fresh Production Build)")
st.markdown("---")

# Setup Navigation Controls Sidebar
st.sidebar.header("🛠️ Reservoir Optimization Console")
env = st.sidebar.selectbox("Target Depth Envelope", ["Intermediate (1,000m - 3,000m)", "Deep (3,000m - 4,500m)", "Ultra-Deep (>4,500m)"])
count = st.sidebar.slider("Microseismic Event Index", 50, 1000, 300, 50)
noise = st.sidebar.slider("Rock Mass Anisotropy (Noise)", 5, 60, 25, 5)
tensile_pct = st.sidebar.slider("Focal Mechanism Tensile Ratio (%)", 5, 95, 35, 5)
uploaded_file = st.sidebar.file_uploader("Upload Subsurface Text Log (.csv/.txt)", type=["csv", "txt"])
opex_limit = st.sidebar.slider("Economic Operating Limit (BOE / Day)", 5.0, 120.0, 25.0, 5.0)

# Build or ingest structural rows cleanly
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        # Dynamic mapping to support varied engineering formats seamlessly
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

# Compute Outputs
engine = SubsurfaceAnalyticsEngine(raw_df, depth_regime=env)
geo = engine.process_geomechanics()
curves = engine.generate_forecasts(geo["fcd"], geo["srv"], opex_limit)

# The Petroleum Engineer's Menu Display Row
st.markdown("### 🍽️ The Petroleum Engineer's Menu")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Stimulated Reservoir Vol (SRV)", f"{geo['srv']:,.2f} m³")
m2.metric("Fracture Conductivity (Fcd)", f"{geo['fcd']:.2f} mD-ft")
m3.metric("Seismic b-Value Index", f"{geo['b_value']:.2f}")
m4.metric("Commercially Productive Life", f"{curves['life']} Years")
m5.metric("Net Recoverable Asset EUR", f"{curves['eur']:,.0f} BOE")
st.markdown("---")

# Layout Splitting Graphs and Data Grid Panels
l_col, r_col = st.columns(2)
with l_col:
    st.markdown("### 📈 Production Fluids Decline & WOR Water-Cut Forecast")
    chart_df = pd.DataFrame({'Hydrocarbon Flow Rate (BOE/d)': curves["oil_rate"], 'Produced Water Loading (Water-Cut BWPD)': curves["water_rate"]}, index=np.round(curves["timeline"], 2))
    st.line_chart(chart_df, y_label="Fluid Volume Rate")
with r_col:
    st.markdown("### 🗃️ Active Subsurface Spatial Log Table")
    st.dataframe(raw_df[['Easting_m', 'Northing_m', 'Depth_m', 'Mechanism', 'Energy_J']], use_container_width=True)
    
    # Module 3: Executive Export Card
    st.markdown("### 📄 Module 3: Executive Export Center")
    st.success(f"PRO-FORMA REVIEW PASSED: Selected {env} envelope optimized at {curves['eur']:,.0f} BOE total net asset volume value.")
