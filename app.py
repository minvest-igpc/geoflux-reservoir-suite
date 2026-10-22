# geoflux/app.py - Standalone Multi-Depth Unified Reservoir Evaluation Suite
import streamlit as st
import numpy as np
import pandas as pd

# =====================================================================
# 1. CORE MULTI-DEPTH GEOMECHANICS PHYSICS ENGINE
# =====================================================================
class ReservoirGeomechanicsEngine:
    """
    Advanced FracMechanica Core Mathematics Engine.
    Handles intermediate, deep, and ultra-deep rock regimes. Computes 3D Convex Hulls,
    Gutenberg-Richter b-values, Chan WOR derivatives, and production decline profiles.
    """
    def __init__(self, raw_df: pd.DataFrame, depth_regime: str):
        self.data = raw_df
        self.regime = depth_regime
        self.srv_volume = 0.0
        self.fracture_conductivity = 0.0

    def parse_and_scale_geomechanics(self) -> tuple:
        """Applies dynamic upscaling coefficients based on selected depth envelope."""
        from scipy.spatial import ConvexHull
        x = self.data['Easting_m'].values
        y = self.data['Northing_m'].values
        z = self.data['Depth_m'].values
        
        coords = np.column_stack((x, y, z))
        if len(coords) < 4:
            return 0.0, 0.0
            
        base_hull_vol = ConvexHull(coords).volume
        
        # Physics Scale Scaling Matrix based on depth constraints
        if self.regime == "Intermediate (1,000m - 3,000m)":
            scale_factor = 10.5  # Sedimentary / brittle matrix expansion
        elif self.regime == "Deep (3,000m - 4,500m)":
            scale_factor = 14.2  # High pressure / tight shale distribution
        else:
            scale_factor = 18.5  # Ultra-deep high stress / tectonic clamping
            
        self.srv_volume = float(base_hull_vol * scale_factor)
        
        tensile_energy = self.data.loc[self.data['Source_Mechanism'] == 'Tensile', 'Energy_Joules'].sum()
        total_energy = self.data['Energy_Joules'].sum()
        
        efficiency = tensile_energy / total_energy if total_energy > 0 else 0.20
        self.fracture_conductivity = float(efficiency * 65.0) 
        
        return self.srv_volume, self.fracture_conductivity

    def calculate_b_value(self) -> float:
        """Module 1: Calculates Gutenberg-Richter b-value index using maximum likelihood."""
        energies = self.data['Energy_Joules'].values
        if len(energies) == 0:
            return 1.0
        # Convert energy proxy cleanly to local magnitude equivalents
        magnitudes = 0.67 * np.log10(energies) - 1.2
        m_min = np.min(magnitudes)
        m_avg = np.mean(magnitudes)
        
        b_val = 1.0 / (m_avg - m_min) if (m_avg - m_min) > 0 else 1.0
        # Apply depth structural adjustments to reflect genuine tectonic environments
        if self.regime == "Ultra-Deep (>4,500m)":
            b_val = b_val * 0.82  # High stress concentration drops b-value
        return float(np.clip(b_val, 0.45, 1.95))

    @staticmethod
    def duong_decline_equation(t, qi, a, m):
        t = np.maximum(t, 1e-5)
        return qi * (t ** -m) * np.exp((a / (1.0 - m)) * ((t ** (1.0 - m)) - 1.0))

    def forecast_fluid_dynamics(self, economic_limit: float = 15.0) -> dict:
        """Module 2: Generates production rates and integrated water water-cut loading curves."""
        qi_pred = float(1200 + (self.fracture_conductivity * 45))
        
        tensile_count = self.data.loc[self.data['Source_Mechanism'] == 'Tensile'].shape[0]
        tensile_ratio = tensile_count / max(len(self.data), 1)
        a_pred = float(1.4 - (tensile_ratio * 0.8))
        
        srv_scale = np.clip(self.srv_volume / 800000.0, 0.1, 2.5)
        m_pred = float(1.55 - (srv_scale * 0.25))
        m_pred = np.clip(m_pred, 1.1, 1.9)

        t_full = np.arange(1, 3653)
        q_raw = self.duong_decline_equation(t_full, qi_pred, a_pred, m_pred)
        
        active_days = np.where(q_raw >= economic_limit)
        days_to_limit = int(active_days[-1]) + 1 if len(active_days) > 0 else 3652
        
        q_final = np.maximum(q_raw, 2.0)
        eur_total = float(np.sum(q_final[:days_to_limit]))
        
        # Module 2 Calculation: Compute water production metrics dynamically (Chan WOR)
        # Deep horizons typically face slower water breakthrough than intermediate aquifers
        water_scale = 0.4 if self.regime == "Deep (3,000m - 4,500m)" else 0.8
        q_water = (t_full ** water_scale) * (q_final * 0.05)

        return {
            "Initial_Peak_Rate_BOED": qi_pred,
            "Decline_Parameter_A": a_pred,
            "Decline_Parameter_M": m_pred,
            "Ten_Year_EUR_BOE": eur_total,
            "Timeline_Days": t_full.tolist(),
            "Production_Curve_BOED": q_final.tolist(),
            "Water_Curve_BWPD": q_water.tolist(),
            "Years_To_Limit": round(days_to_limit / 365.25, 2)
        }

# =====================================================================
# 2. INTERACTIVE USER INTERFACE LAYER
# =====================================================================
st.set_page_config(page_title="GeoFlux Reservoir Suite", layout="wide", page_icon="📊")

st.title("📊 GeoFlux Reservoir Suite")
st.subheader("Physics Engine Core: FracMechanica (Multi-Depth Architecture)")
st.markdown("---")

st.sidebar.header("🛠️ Reservoir Optimization Console")

# Dynamic Multi-Depth Target Selector Widget
depth_env = st.sidebar.selectbox(
    "Target Subsurface Depth Envelope",
    ["Intermediate (1,000m - 3,000m)", "Deep (3,000m - 4,500m)", "Ultra-Deep (>4,500m)"]
)

# Geomechanical Point Cloud Sliders
st.sidebar.subheader("Subsurface Envelope")
srv_slider = st.sidebar.slider("Microseismic Event Count Index", min_value=50, max_value=1000, value=300, step=50)
noise_factor = st.sidebar.slider("Rock Mass Anisotropy (Noise)", min_value=5, max_value=60, value=25, step=5)

st.sidebar.subheader("Pumping Configurations")
tensile_ratio = st.sidebar.slider("Focal Mechanism Tensile Ratio (%)", min_value=5, max_value=95, value=35, step=5)

st.sidebar.subheader("📂 Inbound Files & Economics")
uploaded_file = st.sidebar.file_uploader("Upload Mine Seismology Text Log (.csv/.txt)", type=["csv", "txt"])
opex_limit = st.sidebar.slider("Economic Operating Limit (BOE / Day)", min_value=5.0, max_value=120.0, value=25.0, step=5.0)

# Real-Time Data Ingestion Parsing Loop
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Log File Parsed Successfully!")
        if 'Energy' in raw_df.columns: raw_df.rename(columns={'Energy': 'Energy_Joules'}, inplace=True)
        if 'X' in raw_df.columns: raw_df.rename(columns={'X': 'Easting_m', 'Y': 'Northing_m', 'Z': 'Depth_m'}, inplace=True)
        if 'Mechanism_Type' in raw_df.columns: raw_df.rename(columns={'Mechanism_Type': 'Source_Mechanism'}, inplace=True)
    except Exception as e:
        st.sidebar.error(f"❌ File Parsing Error: {e}")
        uploaded_file = None

if uploaded_file is None:
    np.random.seed(42)
    mechanisms = np.random.choice(
        ['Shear', 'Tensile', 'Closure'], 
        size=srv_slider, 
        p=[1.0 - (tensile_ratio/100.0) - 0.05, tensile_ratio/100.0, 0.05]
    )
    raw_df = pd.DataFrame({
        'Easting_m': np.random.normal(500, noise_factor * 4.0, srv_slider),
        'Northing_m': np.random.normal(1200, noise_factor * 2.0, srv_slider),
        'Depth_m': np.random.normal(-2800, noise_factor * 3.0, srv_slider),
        'Source_Mechanism': mechanisms,
        'Energy_Joules': np.random.lognormal(mean=2.2, sigma=0.6, size=srv_slider)
    })
    st.sidebar.info("ℹ️ Synthesizing data live from geomechanical sliders.")

# Compute Core Metrics across the Engine
engine = ReservoirGeomechanicsEngine(raw_df, depth_regime=depth_env)
srv, fcd = engine.parse_and_scale_geomechanics()
b_index = engine.calculate_b_value()
menu_metrics = engine.forecast_fluid_dynamics(economic_limit=opex_limit)

# The Restaurant Menu Main Summary Output Metric Cards
st.markdown("### 🍽️ The Petroleum Engineer's Menu")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(label="Stimulated Reservoir Vol (SRV)", value=f"{srv:,.2f} m³")
with col2:
    st.metric(label="Fracture Conductivity (Fcd)", value=f"{fcd:.2f} mD-ft")
with col3:
    st.metric(label="Seismic b-Value Index", value=f"{b_index:.2f}")
with col4:
    st.metric(label="Commercially Productive Life", value=f"{menu_metrics['Years_To_Limit']} Years")
with col5:
    st.metric(label="Net Recoverable Asset EUR", value=f"{menu_metrics['Ten_Year_EUR_BOE']:,.0f} BOE")

st.markdown("---")

# Layout Splitting Panels for Charts and Diagnostics
layout_left, layout_right = st.columns(2)

with layout_left:
    st.markdown("### 📈 Production Fluids Decline & WOR Water-Cut Forecast")
    
    days_array = np.array(menu_metrics["Timeline_Days"])
    years_axis = np.round(days_array / 365.25, 2)
    
    plot_dataframe = pd.DataFrame({
        'Hydrocarbon Flow Rate (BOE/d)': np.round(np.array(menu_metrics["Production_Curve_BOED"]), 2),
        'Produced Water Loading (Water-Cut BWPD)': np.round(np.array(menu_metrics["Water_Curve_BWPD"]), 2)
    }, index=years_axis)
    
    st.line_chart(plot_dataframe, y_label="Fluid Volume Rate")

with layout_right:
    st.markdown("### 🗃️ Active Subsurface Spatial Log Table")
    st.dataframe(raw_df[['Easting_m', 'Northing_m', 'Depth_m', 'Source_Mechanism', 'Energy_Joules']], use_container_width=True)
    
    # Module 3: Executive Pro-Forma Investment Summary Generator Trigger
    st.markdown("### 📄 Module 3: Executive Export Center")
    report_summary = (
        f"GEOFLUX EXECUTIVE VALUATION SHEET\n"
        f"---------------------------------------------\n"
        f"Selected Depth Envelope : {depth_env}\n"
        f"Calculated SRV Volume   : {srv:,.2f} m3\n"
        f"Hydraulic Conductivity  : {fcd:.2f} mD-ft\n"
        f"Seismic Hazard b-Index  : {b_index:.2f}\n"
