# geoflux/app.py - Ultimate Combined Standalone Reservoir Suite
import streamlit as st
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

# =====================================================================
# 1. THE FRACMECHANICA GEOMECHANICS PHYSICS ENGINE CLASS
# =====================================================================
class ReservoirGeomechanicsEngine:
    """
    FracMechanica Core Mathematics Engine.
    Processes microseismic data coordinates, computes 3D Convex Hull envelopes, 
    and models production decline profiles matching unconventional reservoir physics.
    """
    def __init__(self, raw_df: pd.DataFrame):
        self.data = raw_df
        self.srv_volume = 0.0
        self.fracture_conductivity = 0.0

    def parse_and_scale_geomechanics(self) -> tuple:
        """Computes 3D geometry volume metrics and fracture conductivity pathways."""
        x = self.data['Easting_m'].values
        y = self.data['Northing_m'].values
        z = self.data['Depth_m'].values
        
        coords = np.column_stack((x, y, z))
        if len(coords) < 4:
            return 0.0, 0.0
            
        # 1. Stimulated Reservoir Volume (SRV) calculation via 3D Convex Hull Enveloping
        base_hull_vol = ConvexHull(coords).volume
        # Reservoir Upscaling Matrix: Scales micro-seismic clusters to structural reservoirs
        self.srv_volume = float(base_hull_vol * 12.5)
        
        # 2. Fracture Flow Pathway Evaluation via Source Mechanism Inversion
        tensile_energy = self.data.loc[self.data['Source_Mechanism'] == 'Tensile', 'Energy_Joules'].sum()
        total_energy = self.data['Energy_Joules'].sum()
        
        efficiency = tensile_energy / total_energy if total_energy > 0 else 0.20
        self.fracture_conductivity = float(efficiency * 65.0) 
        
        return self.srv_volume, self.fracture_conductivity

    @staticmethod
    def duong_decline_equation(t, qi, a, m):
        """Duong DCA equation model for deep tight shale matrices (m > 1)."""
        t = np.maximum(t, 1e-5)
        return qi * (t ** -m) * np.exp((a / (1.0 - m)) * ((t ** (1.0 - m)) - 1.0))

    def forecast_eur(self, economic_limit: float = 15.0) -> dict:
        """Calculates dynamic decline production forecasting curves bounded by economic cut-offs."""
        # Initial flow rate scales dynamically with calculated fracture path conductivity
        qi_pred = float(1200 + (self.fracture_conductivity * 45))
        qi_pred = np.clip(qi_pred, 1000, 6000)
        
        # Slower early decline rate (lower 'a') maps to higher rock tensile cracking density
        tensile_count = self.data.loc[self.data['Source_Mechanism'] == 'Tensile'].shape[0]
        tensile_ratio = tensile_count / max(len(self.data), 1)
        a_pred = float(1.4 - (tensile_ratio * 0.8))
        a_pred = np.clip(a_pred, 0.3, 1.5)
        
        # Larger drainage volume (SRV) extends the life of the decline tail (lower 'm')
        srv_scale = np.clip(self.srv_volume / 800000.0, 0.1, 2.5)
        m_pred = float(1.55 - (srv_scale * 0.25))
        m_pred = np.clip(m_pred, 1.1, 1.9)

        # Build 10-Year time integration timeline array (3652 days)
        t_full = np.arange(1, 3653)
        q_raw = self.duong_decline_equation(t_full, qi_pred, a_pred, m_pred)
        
        # Calculate exactly when daily rates drop below profitability boundaries
        active_days = np.where(q_raw >= economic_limit)[0]
        days_to_limit = int(active_days[-1]) + 1 if len(active_days) > 0 else 3652
        
        q_final = np.maximum(q_raw, 2.0)
        eur_total = float(np.sum(q_final[:days_to_limit]))

        return {
            "Initial_Peak_Rate_BOED": qi_pred,
            "Decline_Parameter_A": a_pred,
            "Decline_Parameter_M": m_pred,
            "Ten_Year_EUR_BOE": eur_total,
            "Timeline_Days": t_full.tolist(),
            "Production_Curve_BOED": q_final.tolist(),
            "Years_To_Limit": round(days_to_limit / 365.25, 2)
        }

# =====================================================================
# 2. DYNAMIC INTERACTIVE USER INTERFACE LAYER
# =====================================================================
st.set_page_config(page_title="GeoFlux Reservoir Suite", layout="wide", page_icon="📊")

st.title("📊 GeoFlux Reservoir Suite")
st.subheader("Physics Engine Core: FracMechanica (Complete Integrated Architecture)")
st.markdown("---")

# 1. RESTORE ALL ORIGINAL GEOMECHANICAL AND DATA CONTROLS IN THE SIDEBAR
st.sidebar.header("🛠️ Reservoir Optimization Console")

# --- Category 1: Original Subsurface Point Cloud Sliders ---
st.sidebar.subheader("Subsurface Envelope")
srv_slider = st.sidebar.slider("Microseismic Event Count Index", min_value=50, max_value=1000, value=300, step=50)
noise_factor = st.sidebar.slider("Rock Mass Anisotropy (Noise)", min_value=5, max_value=60, value=25, step=5)

# --- Category 2: Original Pumping Configuration Sliders ---
st.sidebar.subheader("Pumping Configurations")
tensile_ratio = st.sidebar.slider("Focal Mechanism Tensile Ratio (%)", min_value=5, max_value=95, value=35, step=5)

# --- Category 3: New Data Logs & OPEX Financial Blocks ---
st.sidebar.subheader("📂 Inbound Files & Economics")
uploaded_file = st.sidebar.file_uploader("Upload Mine Seismology Text Log (.csv/.txt)", type=["csv", "txt"])
opex_limit = st.sidebar.slider("Economic Operating Limit (BOE / Day)", min_value=5.0, max_value=120.0, value=25.0, step=5.0)

# 2. RUN REAL-TIME DATA COMPILATION ROUTINES
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Log File Parsed Successfully!")
        # Rename incoming external variations on the fly to fit variables
        if 'Energy' in raw_df.columns: raw_df.rename(columns={'Energy': 'Energy_Joules'}, inplace=True)
        if 'X' in raw_df.columns: raw_df.rename(columns={'X': 'Easting_m', 'Y': 'Northing_m', 'Z': 'Depth_m'}, inplace=True)
        if 'Mechanism_Type' in raw_df.columns: raw_df.rename(columns={'Mechanism_Type': 'Source_Mechanism'}, inplace=True)
    except Exception as e:
        st.sidebar.error(f"❌ File Parsing Error: {e}")
        uploaded_file = None

if uploaded_file is None:
    # RE-INTEGRATE THE ORIGINAL MATHEMATICAL STOCHASTIC SIMULATOR
    # Point cloud scales dynamically right off your original sliders in real time
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

# Fire calculations dynamically across the newly synchronized platform
engine = ReservoirGeomechanicsEngine(raw_df)
srv, fcd = engine.parse_and_scale_geomechanics()
menu_metrics = engine.forecast_eur(economic_limit=opex_limit)

# 3. THE RESTAURANT MENU VIEW (Top Analytics Output Display Metrics Row)
st.markdown("### 🍽️ The Petroleum Engineer's Menu")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Stimulated Reservoir Volume (SRV)", value=f"{srv:,.2f} m³")
with col2:
    st.metric(label="Fracture Conductivity (Fcd)", value=f"{fcd:.2f} mD-ft")
with col3:
    st.metric(label="Commercially Productive Life", value=f"{menu_metrics['Years_To_Limit']} Years")
with col4:
    st.metric(label="Net Recoverable Asset EUR", value=f"{menu_metrics['Ten_Year_EUR_BOE']:,.0f} BOE")

st.markdown("---")

# 4. SPLIT SIDE-BY-SIDE ANALYTICAL INTERACTIVE GRAPHIC PANELS
layout_left, layout_right = st.columns(2)

with layout_left:
    st.markdown("### 📈 Hyperbolic Asset Decline Drawdown (Sweeping Profile)")
    
    days_array = np.array(menu_metrics["Timeline_Days"])
    years_axis = np.round(days_array / 365.25, 2)
    production_rates = np.round(np.array(menu_metrics["Production_Curve_BOED"]), 2)
    
    plot_dataframe = pd.DataFrame({
        'Daily Output (BOE/d)': production_rates
    }, index=years_axis)
    
    # Ultra-stable line chart visualization engine
    st.line_chart(plot_dataframe, y_label="Production Rate (BOE / Day)")

with layout_right:
    st.markdown("### 🗃️ Active Subsurface Spatial Log Table")
    # Display running structural entries table grid layout
    st.dataframe(raw_df[['Easting_m', 'Northing_m', 'Depth_m', 'Source_Mechanism', 'Energy_Joules']], use_container_width=True)
    
    st.json({
        "Model_Fitted_Decline_Parameters": {
            "Initial_Peak_Rate_BOED": round(menu_metrics["Initial_Peak_Rate_BOED"], 2),
            "Constant_Decline_A": round(menu_metrics["Decline_Parameter_A"], 4),
            "Constant_Decline_M": round(menu_metrics["Decline_Parameter_M"], 4)
        },
        "Geomechanical_Cloud_Diagnostics": {
            "Total_Parsed_Microseismic_Points": len(raw_df),
            "Tensile_Opening_Events_Count": int(np.sum(raw_df['Source_Mechanism'] == 'Tensile')),
            "Total_Log_Energy_Joules": round(float(raw_df['Energy_Joules'].sum()), 2)
        }
    })
