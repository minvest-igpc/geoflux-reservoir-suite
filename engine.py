import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

class ReservoirGeomechanicsEngine:
    """
    FracMechanica Core Engine.
    Parses deep-level mining microseismic source logs, applies reservoir upscaling 
    matrices, and models long-term EUR production dynamics.
    """
    def __init__(self, raw_df: pd.DataFrame):
        self.data = raw_df
        self.srv_volume = 0.0
        self.fracture_conductivity = 0.0

    def parse_and_scale_geomechanics(self) -> tuple:
        """
        Upscales micro-scale mine parameters to macroscopic oilfield reservoir scale.
        """
        x = self.data['Easting_m'].values
        y = self.data['Northing_m'].values
        z = self.data['Depth_m'].values
        
        coords = np.column_stack((x, y, z))
        if len(coords) < 4:
            return 0.0, 0.0
            
        # 1. Volumetric Expansion Metric via 3D Convex Hull Enveloping
        base_hull_vol = ConvexHull(coords).volume
        
        # Reservoir Upscaling Matrix (~x10 scale boost)
        self.srv_volume = float(base_hull_vol * 10.5)
        
        # 2. Fracture Flow Pathway Evaluation via Source Mechanism Inversion
        tensile_energy = self.data.loc[self.data['Source_Mechanism'] == 'Tensile', 'Energy_Joules'].sum()
        total_energy = self.data['Energy_Joules'].sum()
        
        efficiency_factor = tensile_energy / total_energy if total_energy > 0 else 0.1
        self.fracture_conductivity = float(efficiency_factor * 65.0) 
        
        return self.srv_volume, self.fracture_conductivity

    @staticmethod
    def duong_decline_equation(t, qi, a, m):
        t = np.maximum(t, 1e-5)
        return qi * (t ** -m) * np.exp((a / (1.0 - m)) * ((t ** (1.0 - m)) - 1.0))

    def forecast_eur(self, economic_limit: float = 15.0) -> dict:
        """
        Generates production curves under specific economic cut-offs.
        """
        qi_pred = float(1200 + (self.fracture_conductivity * 40))
        
        tensile_count = self.data.loc[self.data['Source_Mechanism'] == 'Tensile'].shape[0]
        tensile_ratio = tensile_count / max(len(self.data), 1)
        a_pred = float(1.3 - (tensile_ratio * 0.7))
        
        srv_scale = np.clip(self.srv_volume / 500000.0, 0.2, 2.5)
        m_pred = float(1.45 - (srv_scale * 0.15))
        m_pred = np.clip(m_pred, 1.05, 1.8)

        t_full = np.arange(1, 3653)
        q_raw = self.duong_decline_equation(t_full, qi_pred, a_pred, m_pred)
        
        active_days = np.where(q_raw >= economic_limit)[0]
        days_to_economic_limit = int(active_days[-1]) + 1 if len(active_days) > 0 else 3652
        
        q_final = np.maximum(q_raw, 2.0)
        eur_total = float(np.sum(q_final[:days_to_economic_limit]))

        return {
            "Initial_Peak_Rate_BOED": qi_pred,
            "Decline_Parameter_A": a_pred,
            "Decline_Parameter_M": m_pred,
            "Ten_Year_EUR_BOE": eur_total,
            "Timeline_Days": t_full.tolist(),
            "Production_Curve_BOED": q_final.tolist(),
            "Days_To_Limit": days_to_economic_limit,
            "Years_To_Limit": round(days_to_economic_limit / 365.25, 2)
        }
