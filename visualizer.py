import os
import matplotlib.pyplot as plt
import numpy as np

def generate_production_plot(eur_results: dict, output_dir: str = "exports/petroleum_menu"):
    """
    Generates a publication-grade reservoir engineering decline curve.
    Outputs directly to the client-facing 'Petroleum Menu' folder.
    """
    # Ensure the export directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract times and rates from the engine forecast
    days = np.array(eur_results["Timeline_Days"])
    years = days / 365.0
    rates = np.array(eur_results["Production_Curve_BOED"])
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Plot the decline curve
    ax.plot(years, rates, color='#006699', linewidth=2.5, label='Forecasted Decline (Duong Model)')
    
    # Shade the area under the curve (This physically represents the cumulative EUR)
    ax.fill_between(years, rates, color='#006699', alpha=0.1, label='Cumulative EUR Volume')
    
    # Titles & Labels customized for oilfield asset management
    ax.set_title("GeoFlux Asset Forecast: 10-Year Estimated Ultimate Recovery", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Time (Years)", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Production Rate (BOE / Day)", fontsize=11, fontweight='semibold')
    
    # Set axes limits precisely to prevent layout clipping
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(rates) * 1.1)
    
    # Format gridlines to look sleek and non-intrusive
    ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    
    # Add a clean text information box highlighting key performance indicators (KPIs)
    kpi_text = (
        f"Initial Peak (qi): {eur_results['Initial_Peak_Rate_BOED']:,.2f} BOE/d\n"
        f"Decline Param (a): {eur_results['Decline_Parameter_A']:.4f}\n"
        f"Decline Param (m): {eur_results['Decline_Parameter_M']:.4f}\n"
        f"---------------------------\n"
        f"TOTAL 10-YR EUR: {eur_results['Ten_Year_EUR_BOE']:,.0f} BOE"
    )
    
    props = dict(boxstyle='round,pad=0.8', facecolor='#ffffff', edgecolor='#bbbbbb', alpha=0.9)
    ax.text(0.95, 0.95, kpi_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props, fontfamily='monospace')
    
    ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#eeeeee')
    
    # Save chart cleanly with tight borders
    output_path = os.path.join(output_dir, "eur_decline_curve.png")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"[GeoFlux] Analytics graph successfully rendered and saved to: {output_path}")
