# geoflux/app.py - Ultimate Combined Standalone Reservoir Suite
import streamlit as st
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

# =====================================================================
# SPATIOTEMPORAL NOISE FILTER FOR MINE-INDUCED EVENTS
# =====================================================================
def filter_seismic_noise(
    seismic_df: pd.DataFrame, 
    mine_geometry_df: pd.DataFrame, 
    max_dist_meters: float = 30.0,
    time_window_days: float = 7.0
) -> pd.DataFrame:
    """Filters out mine-induced background noise from microseismic data logs."""
    required_cols = ['Easting_m', 'Northing_m', 'Depth_m', 'timestamp']
    for col in required_cols:
        if col not in seismic_df.columns:
            if col == 'timestamp':
                seismic_df['timestamp'] = pd.Timestamp.now()
            else:
                return seismic_df
                
    if mine_geometry_df is None or mine_geometry_df.empty:
        return seismic_df

    seismic_df['timestamp'] = pd.to_datetime(seismic_df['timestamp'])
    mine_geometry_df['active_timestamp'] = pd.to_datetime(mine_geometry_df['active_timestamp'])
    
    noise_indices = set()
    for _, mine_node in mine_geometry_df.iterrows():
        time_diffs = np.abs((seismic_df['timestamp'] - mine_node['active_timestamp']).dt.days)
        time_mask = time_diffs <= time_window_days
        if not time_mask.any():
            continue
            
        candidates = seismic_df[time_mask]
        dx = candidates['Easting_m'] - mine_node['x']
        dy = candidates['Northing_m'] - mine_node['y']
        dz = candidates['Depth_m'] - mine_node['z']
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        
        spatial_mask = distances <= max_dist_meters
        flagged_noise = candidates[spatial_mask].index
        noise_indices.update(flagged_noise)
        
    return seismic_df.drop(index=list(noise_indices))

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
            
        # Stimulated Reservoir Volume (SRV) calculation via 3D Convex Hull Enveloping
        base_hull_vol = ConvexHull(coords).volume
        self.srv_volume = float(base_hull_vol * 12.5)
        
        # Fracture Flow Pathway Evaluation via Source Mechanism Inversion
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
        qi_pred = float(1200 + (self.fracture_conductivity * 45))
        qi_pred = np.clip(qi_pred, 1000, 6000)
        
        # Extracted integer indices from layout shape matrices safely
        tensile_count = int(self.data.loc[self.data['Source_Mechanism'] == 'Tensile'].shape[0])
        tensile_ratio = tensile_count / max(len(self.data), 1)
        a_pred = float(1.4 - (tensile_ratio * 0.8))
        a_pred = np.clip(a_pred, 0.3, 1.5)
        
        srv_scale = np.clip(self.srv_volume / 800000.0, 0.1, 2.5)
        m_pred = float(1.55 - (srv_scale * 0.25))
        m_pred = np.clip(m_pred, 1.1, 1.9)

        t_full = np.arange(1, 3653)
        q_raw = self.duong_decline_equation(t_full, qi_pred, a_pred, m_pred)
        
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

st.sidebar.header("🛠️ Reservoir Optimization Console")

# Subsurface Envelope Sliders
st.sidebar.subheader("Subsurface Envelope")
srv_slider = st.sidebar.slider("Microseismic Event Count Index", min_value=50, max_value=1000, value=300, step=50)
noise_factor = st.sidebar.slider("Rock Mass Anisotropy (Noise)", min_value=5, max_value=60, value=25, step=5)

# Spatiotemporal Multi-Stream Filter Controls
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Mine Seismology Noise Clean-off")
enable_filter = st.sidebar.checkbox("Activate Spatiotemporal Buffer", value=True)
buffer_dist = st.sidebar.slider("Spatial Exclusion Radius (Meters)", min_value=5.0, max_value=100.0, value=30.0, step=5.0)
buffer_time = st.sidebar.slider("Temporal Operation Window (Days)", min_value=1.0, max_value=30.0, value=7.0, step=1.0)
st.sidebar.markdown("---")

# Pumping Configuration Sliders
st.sidebar.subheader("Pumping Configurations")
tensile_ratio = st.sidebar.slider("Focal Mechanism Tensile Ratio (%)", min_value=5, max_value=95, value=35, step=5)

# Inbound Files & Economics Blocks
st.sidebar.subheader("📂 Inbound Files & Economics")
uploaded_file = st.sidebar.file_uploader("Upload Mine Seismology Text Log (.csv/.txt)", type=["csv", "txt"])
opex_limit = st.sidebar.slider("Economic Operating Limit (BOE / Day)", min_value=5.0, max_value=120.0, value=25.0, step=5.0)

# Real-Time Data Compilation Pipeline
raw_df = None

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Log File Parsed Successfully!")
        if 'Energy' in raw_df.columns: raw_df.rename(columns={'Energy': 'Energy_Joules'}, inplace=True)
        if 'X' in raw_df.columns: raw_df.rename(columns={'X': 'Easting_m', 'Y': 'Northing_m', 'Z': 'Depth_m'}, inplace=True)
        if 'Mechanism_Type' in raw_df.columns: raw_df.rename(columns={'Mechanism_Type': 'Source_Mechanism'}, inplace=True)
        if 'Time' in raw_df.columns: raw_df.rename(columns={'Time': 'timestamp'}, inplace=True)
    except Exception as e:
        st.sidebar.error(f"❌ File Parsing Error: {e}")
        raw_df = None

if raw_df is None:
    np.random.seed(42)
    mechanisms = np.random.choice(
        ['Shear', 'Tensile', 'Closure'], 
        size=srv_slider, 
        p=[1.0 - (tensile_ratio/100.0) - 0.05, tensile_ratio/100.0, 0.05]
    )
    base_dates = pd.date_range(end=pd.Timestamp.now(), periods=srv_slider, freq='h')
    
    raw_df = pd.DataFrame({
        'Easting_m': np.random.normal(500, noise_factor * 4.0, srv_slider),
        'Northing_m': np.random.normal(1200, noise_factor * 2.0, srv_slider),
        'Depth_m': np.random.normal(-2800, noise_factor * 3.0, srv_slider),
        'Source_Mechanism': mechanisms,
        'Energy_Joules': np.random.lognormal(mean=2.2, sigma=0.6, size=srv_slider),
        'timestamp': base_dates
    })
    st.sidebar.info("ℹ️ Synthesizing data live from geomechanical sliders.")

# Execute Data Filtering Process
if enable_filter:
    mock_mine_geometry = pd.DataFrame({
        'mine_zone_id': ['Zone_A_Longwall', 'Zone_B_Pillar'],
        'x': [490.0, 520.0],
        'y': [1180.0, 1210.0],
        'z': [-2790.0, -2810.0],
        'active_timestamp': [pd.Timestamp.now(), pd.Timestamp.now() - pd.Timedelta(days=2)]
    })
    
    initial_count = len(raw_df)
    raw_df = filter_seismic_noise(raw_df, mock_mine_geometry, max_dist_meters=buffer_dist, time_window_days=buffer_time)
    filtered_out = initial_count - len(raw_df)
    if filtered_out > 0:
        st.sidebar.warning(f"⚡ Isolated {filtered_out} structural mine-noise events from logs.")

# Compute System Metrics
engine = ReservoirGeomechanicsEngine(raw_df)
srv, fcd = engine.parse_and_scale_geomechanics()
menu_metrics = engine.forecast_eur(economic_limit=opex_limit)

# Metrics Grid Section
st.markdown("### 🍽️ The Petroleum Engineer's Menu")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Stimulated Reservoir Volume (SRV)", value=f"{srv:,.2f} m³")
with col2:
    st.metric(label="Fracture Conductivity (Fcd)", value=f"{fcd:.2f} mD-ft")
with col3:
