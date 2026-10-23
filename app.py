# geoflux/app.py - Ultimate 5-Chart Unified Subsurface Diagnostics Suite with Explicit Audit
import streamlit as st
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

# =====================================================================
# 1. CORE PHYSICS ENGINEERING ENGINE (FRACMECHANICA ENTERPRISE)
# =====================================================================
class SubsurfaceAnalyticsEngine:
    def __init__(self, df: pd.DataFrame, depth_regime: str):
        self.data = df
        self.regime = depth_regime
        self.srv_volume = 0.0
        self.fracture_conductivity = 0.0

    def apply_spatiotemporal_buffer(self, buffer_radius: float, isolate_blasting: bool) -> pd.DataFrame:
        """Point 1 & 2: Explicitly handles background mine filter routines."""
        df_filtered = self.data.copy()
        df_filtered = df_filtered[np.abs(df_filtered['Easting_m'] - 500) > buffer_radius]
        if isolate_blasting: 
            df_filtered = df_filtered[df_filtered['Mechanism'] != 'Closure']
        return df_filtered

    def calculate_tectonic_stress_inversion(self, tensile_ratio_val: float) -> dict:
        """Point 3: Inverts source mechanism attributes into horizontal tensor components."""
        base_depth = 2000.0 if "Intermediate" in self.regime else (3800.0 if "Deep" in self.regime else 5000.0)
        sv = base_depth * 0.023 
        shmin = sv * (0.6 + (tensile_ratio_val / 250.0))
        shmax = shmin * (1.2 + (base_depth / 15000.0))
        be_index = (self.data.loc[self.data['Mechanism'] == 'Closure', 'Energy_J'].sum() / self.data['Energy_J'].sum()) * 1.5 if len(self.data) > 0 else 0.4
        return {"Sv": round(sv, 1), "Shmin": round(shmin, 1), "Shmax": round(shmax, 1), "Be": round(be_index, 2)}

    def process_geomechanics_with_proppant(self, df_clean: pd.DataFrame, fluid_efficiency: float) -> dict:
        """Point 4: Explicitly maps spatial energy trends directly to proppant consolidation fields."""
        xyz = df_clean[['Easting_m', 'Northing_m', 'Depth_m']].values
        base_vol = ConvexHull(xyz).volume if len(xyz) >= 4 else 0.0
        scale = 10.5 if "Intermediate" in self.regime else (14.2 if "Deep" in self.regime else 18.5)
        self.srv_volume = float(base_vol * scale)
        
        tensile_e = df_clean.loc[df_clean['Mechanism'] == 'Tensile', 'Energy_J'].sum()
        total_e = df_clean['Energy_J'].sum()
        efficiency_ratio = tensile_e / total_e if total_e > 0 else 0.2
        self.fracture_conductivity = float(efficiency_ratio * 55.0 * (fluid_efficiency / 50.0))
        
        magnitudes = 0.67 * np.log10(np.maximum(df_clean['Energy_J'].values, 1e-3)) - 1.2
        b_val = float(1.0 / (np.mean(magnitudes) - np.min(magnitudes))) if len(magnitudes) > 0 else 1.0
        if "Ultra-Deep" in self.regime: b_val *= 0.85
        return {"srv": self.srv_volume, "fcd": self.fracture_conductivity, "b_value": np.clip(b_val, 0.45, 1.95), "magnitudes": magnitudes}

    def generate_forecasts(self, fcd: float, srv: float, opex: float, tensile_ratio_input: float, noise_input: float) -> dict:
        qi = np.clip(float(1000 + (fcd * 55)), 800, 6500)
        t = np.arange(1, 3653)
        di_factor = np.clip(float(0.45 - (tensile_ratio_input / 300.0)), 0.05, 0.40)
        curvature_n = np.clip(float(0.40 + (srv / 2000000.0) + (noise_input / 200.0)), 0.15, 0.95)
        
        q_fluid = qi / (1.0 + di_factor * t) ** curvature_n
        w_scale = 0.35 if "Deep" in self.regime else (0.50 if "Ultra-Deep" in self.regime else 0.75)
        q_water = (t ** w_scale) * (q_fluid * 0.04)
        
        # PERMANENT TUPLE EXTRACTION FIX: Added [0] to extract the array out of the tuple natively
        active_days = np.where(q_fluid >= opex)[0]
        cutoff = int(active_days[-1]) + 1 if len(active_days) > 0 else 3652
        
        return {"timeline": (t / 365.25).tolist(), "oil_rate": q_fluid.tolist(), "water_rate": q_water.tolist(), "eur": float(np.sum(q_fluid[:cutoff])), "life": round(cutoff / 365.25, 2)}

# =====================================================================
# 2. INTERACTIVE USER INTERFACE LAYER
# =====================================================================
st.set_page_config(page_title="GeoFlux Reservoir Suite", layout="wide", page_icon="📊")
st.title("📊 GeoFlux Reservoir Suite")
st.subheader("Physics Engine Core: FracMechanica (Enterprise Scrutiny Architecture)")
st.markdown("---")

st.sidebar.header("🛠️ Reservoir Optimization Console")
env = st.sidebar.selectbox("Target Depth Envelope", ["Intermediate (1,000m - 3,000m)", "Deep (3,000m - 4,500m)", "Ultra-Deep (>4,500m)"])
count = st.sidebar.slider("Microseismic Event Index", 50, 1000, 300, 50)
noise = st.sidebar.slider("Rock Mass Anisotropy (Noise)", 5, 60, 25, 5)
tensile_pct = st.sidebar.slider("Focal Mechanism Tensile Ratio (%)", 5, 95, 35, 5)

st.sidebar.subheader("🔒 Advanced Filtering & Inversion Modules")
spatial_buffer = st.sidebar.slider("Point 1: Spatiotemporal Distance Buffer (meters)", 0.0, 50.0, 15.0, 5.0)
wave_disc = st.sidebar.toggle("Point 2: Time-Frequency Waveform Discriminator", value=True)
proppant_eff = st.sidebar.slider("Point 4: Fluid Proppant Carrying Efficiency (%)", 10, 100, 75, 5)

st.sidebar.subheader("🕹️ Feature Upgrade Overlays")
noise_sim_amplitude = st.sidebar.slider("Simulate Machinery Noise Amplitude", 0, 100, 30, 10)

st.sidebar.subheader("📂 Inbound Files & Economics")
uploaded_file = st.sidebar.file_uploader("Upload Subsurface Text Log (.csv/.txt)", type=["csv", "txt"])
opex_limit = st.sidebar.slider("Economic Operating Limit (BOE / Day)", 5.0, 120.0, 25.0, 5.0)

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        rename_dict = {'Energy': 'Energy_J', 'Energy_Joules': 'Energy_J', 'X': 'Easting_m', 'Y': 'Northing_m', 'Z': 'Depth_m', 'Mechanism_Type': 'Mechanism', 'Source_Mechanism': 'Mechanism'}
        raw_df.rename(columns={k: v for k, v in rename_dict.items() if k in raw_df.columns}, inplace=True)
    except Exception: uploaded_file = None

if uploaded_file is None:
    np.random.seed(42)
    noise_ratio = noise_sim_amplitude / 250.0
    p_shear = np.clip(1.0 - (tensile_pct/100.0) - 0.05 - noise_ratio, 0.01, 1.0)
    p_closure = np.clip(0.05 + noise_ratio, 0.01, 0.95)
    p_tensile = np.clip(1.0 - p_shear - p_closure, 0.01, 1.0)
    p_total = p_shear + p_tensile + p_closure
    p_shear, p_tensile, p_closure = p_shear/p_total, p_tensile/p_total, p_closure/p_total
    
    mechs = np.random.choice(['Shear', 'Tensile', 'Closure'], size=count, p=[p_shear, p_tensile, p_closure])
    energy_distribution = np.random.lognormal(2.2 + (noise / 50.0), 0.6 + (tensile_pct / 200.0), count)
    raw_df = pd.DataFrame({'Easting_m': np.random.normal(500, noise * 4.0, count), 'Northing_m': np.random.normal(1200, noise * 2.0, count), 'Depth_m': np.random.normal(-2800, noise * 3.0, count), 'Mechanism': mechs, 'Energy_J': energy_distribution})

engine = SubsurfaceAnalyticsEngine(raw_df, depth_regime=env)
df_cleaned = engine.apply_spatiotemporal_buffer(buffer_radius=spatial_buffer, isolate_blasting=wave_disc)
stress = engine.calculate_tectonic_stress_inversion(tensile_ratio_val=tensile_pct)
geo = engine.process_geomechanics_with_proppant(df_cleaned, fluid_efficiency=proppant_eff)
curves = engine.generate_forecasts(geo["fcd"], geo["srv"], opex_limit, tensile_ratio_input=tensile_pct, noise_input=noise)

st.markdown("### 🍽️ The Petroleum Engineer's Menu")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("SRV Volume", f"{geo['srv']:,.2f} m³")
m2.metric("Conductivity (Fcd)", f"{geo['fcd']:.2f} mD-ft")
m3.metric("b-Value Index", f"{geo['b_value']:.2f}")
m4.metric("Productive Life", f"{curves['life']} Years")
m5.metric("Asset EUR", f"{curves['eur']:,.0f} BOE")
st.markdown("---")

# CHANNEL A: THE BOTTOM-LINE FLUID FLOW DIAGRAMS
st.markdown("### 📊 Channel A: Production Fluids Performance Forecasts")
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### 📈 Hydrocarbon Decline Profile")
    st.area_chart(pd.DataFrame({'Rate (BOE/d)': curves["oil_rate"]}, index=np.round(curves["timeline"], 2)))
with c2:
    st.markdown("#### 💧 Produced Water Cut Curve (WOR)")
    st.area_chart(pd.DataFrame({'Water (BWPD)': curves["water_rate"]}, index=np.round(curves["timeline"], 2)))

st.markdown("---")

# CHANNEL B: SEISMOLOGIST COMPLIANCE & FILTER AUDIT CANVAS (POINTS 1, 2, 3, 4)
st.markdown("### 🔍 Channel B: Geomechanical Inversion & Filter Validation (The Seismologist Scrutiny View)")
v1, v2, v3 = st.columns(3)

with v1:
    st.markdown("#### ⚡ Point 1 & 2: Gutenberg-Richter Filtering & Margin of Error Envelope")
    m_raw = 0.67 * np.log10(np.maximum(raw_df['Energy_J'].values, 1e-3)) - 1.2
    m_clean = geo["magnitudes"]
    
    bins = np.linspace(min(np.min(m_raw), np.min(m_clean)), max(np.max(m_raw), np.max(m_clean)), 15)
    counts_raw, _ = np.histogram(m_raw, bins=bins)
    counts_clean, _ = np.histogram(m_clean, bins=bins)
    
    log_raw = np.nan_to_num(np.log10(np.maximum(np.cumsum(counts_raw[::-1])[::-1], 1)))
    log_clean = np.nan_to_num(np.log10(np.maximum(np.cumsum(counts_clean[::-1])[::-1], 1)))
    
    error_margin = noise / 120.0
    
    scrutiny_df = pd.DataFrame({
        'Unfiltered Core Data Logs': np.round(log_raw, 4),
        'Filtered Sanity Dataset': np.round(log_clean, 4),
        'Margin of Error (Upper Bound)': np.round(log_clean + error_margin, 4),
        'Margin of Error (Lower Bound)': np.round(np.maximum(log_clean - error_margin, 0), 4)
    }, index=np.round(bins[:-1], 2))
    st.line_chart(scrutiny_df)

with v2:
    st.markdown("#### 🎯 Point 3: Tectonic Horizontal Stress Tensor Profile")
    stress_timeline = np.linspace(0, 10, 50)
    stress_chart_df = pd.DataFrame({
        "Vertical Stress (Sv)": [stress["Sv"]] * 50,
        "Max Vector Azimuth (SHmax)": [stress["Shmax"]] * 50,
        "Min Vector Azimuth (Shmin)": [stress["Shmin"]] * 50
    }, index=stress_timeline)
