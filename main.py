import numpy as np
import pandas as pd
import json
from src.frac_mechanica.engine import ReservoirGeomechanicsEngine
from src.frac_mechanica.visualizer import generate_production_plot

def generate_synthetic_mining_data():
    """Simulates raw source data exporting from mine monitoring software."""
    np.random.seed(42)
    events = 300
    return pd.DataFrame({
        'Event_ID': range(1, events + 1),
        'X': np.random.normal(500, 40, events),
        'Y': np.random.normal(1200, 12, events),
        'Z': np.random.normal(-2800, 20, events),
        'Mechanism_Type': np.random.choice(['Shear', 'Tensile', 'Closure'], size=events, p=[0.65, 0.30, 0.05]),
        'Energy': np.random.lognormal(mean=2.1, sigma=0.7, size=events)
    })

def main():
    print("[GeoFlux] Initializing pipeline...")
    
    # 1. Ingest raw microseismic data
    raw_data = generate_synthetic_mining_data()
    
    # 2. Execute the FracMechanica kitchen calculations
    print("[GeoFlux] Invoking FracMechanica core geomechanics engine...")
    engine = ReservoirGeomechanicsEngine(raw_data)
    
    srv = engine.calculate_srv()
    fcd = engine.calculate_fracture_conductivity()
    menu_metrics = engine.forecast_eur()
    
    # 3. Trigger the visualization generator
    print("[GeoFlux] Generating client deliverables...")
    generate_production_plot(menu_metrics)
    
    # 4. Package the final clean Petroleum Menu JSON Report
    petroleum_report = {
        "Well_Evaluation_Summary": {
            "Stimulated_Reservoir_Volume_m3": round(srv, 2),
            "Fracture_Conductivity_mD_ft": round(fcd, 2),
            "Predicted_Initial_Peak_BOED": round(menu_metrics["Initial_Peak_Rate_BOED"], 2),
            "Asset_Value_10Yr_EUR_BOE": round(menu_metrics["Ten_Year_EUR_BOE"], 2)
        }
    }
    
    report_path = 'exports/petroleum_menu/eur_asset_report.json'
    with open(report_path, 'w') as f:
        json.dump(petroleum_report, f, indent=4)
        
    print("\n" + "="*45)
    print("        GEOFLUX PETROLEUM REPORT EXPORTED      ")
    print("="*45)
    print(f" SRV Geometry         : {srv:,.2f} m³")
    print(f" Frac Conductivity    : {fcd:.2f} mD-ft")
    print(f" Target 10-Year EUR   : {menu_metrics['Ten_Year_EUR_BOE']:,.2f} BOE")
    print(f" Report Saved To      : {report_path}")
    print("="*45)

if __name__ == "__main__":
    main()
