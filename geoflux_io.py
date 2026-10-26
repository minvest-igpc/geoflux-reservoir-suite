"""GeoFlux oil & gas: loaders for the treatment stage log and production history."""
import pandas as pd

import geoflux_mine as mine

BBL_TO_M3 = 0.158987294928
PSI_TO_MPA = 0.00689475729

_STAGE_ALIASES = {
    "stage": "Stage", "stage_no": "Stage", "stage_number": "Stage", "stage_id": "Stage",
    "start": "Start", "start_time": "Start", "pump_start": "Start",
    "end": "End", "end_time": "End", "pump_end": "End",
    "volume_m3": "Volume_m3", "vol_m3": "Volume_m3", "fluid_m3": "Volume_m3",
    "volume_bbl": "Volume_bbl", "vol_bbl": "Volume_bbl", "fluid_bbl": "Volume_bbl",
    "max_pressure_mpa": "Max_Pressure_MPa", "pressure_mpa": "Max_Pressure_MPa",
    "max_pressure_psi": "Max_Pressure_psi", "pressure_psi": "Max_Pressure_psi",
}
_PROD_ALIASES = {
    "date": "Date", "month": "Date", "time": "Date", "period": "Date",
    "rate_boed": "Rate_boed", "boed": "Rate_boed", "boe_d": "Rate_boed",
    "boepd": "Rate_boed", "rate": "Rate_boed", "production": "Rate_boed",
}


def load_treatment_log(file):
    """Stage log. Volume must have explicit units (Volume_m3 or Volume_bbl)."""
    warnings = []
    try:
        df = mine._read(file)
    except Exception as exc:
        return None, f"Could not read the treatment log: {exc}", warnings
    df = mine._normalise(df, _STAGE_ALIASES)
    if df.columns.duplicated().any():
        return None, "Two columns map to the same field; rename or remove one.", warnings
    missing = [c for c in ["Stage", "Start", "End"] if c not in df.columns]
    if missing:
        return None, f"Treatment log missing columns: {missing}. Found: {list(df.columns)}", warnings
    if "Volume_m3" not in df.columns and "Volume_bbl" not in df.columns:
        return None, ("Need a volume column with explicit units: Volume_m3 or Volume_bbl "
                      "(a bare 'Volume' column is ambiguous)."), warnings
    if "Volume_m3" not in df.columns:
        df["Volume_m3"] = pd.to_numeric(df["Volume_bbl"], errors="coerce") * BBL_TO_M3
        warnings.append("Volume converted from barrels to cubic metres.")
    df["Volume_m3"] = pd.to_numeric(df["Volume_m3"], errors="coerce")
    if "Max_Pressure_MPa" not in df.columns and "Max_Pressure_psi" in df.columns:
        df["Max_Pressure_MPa"] = pd.to_numeric(df["Max_Pressure_psi"], errors="coerce") * PSI_TO_MPA
        warnings.append("Pressure converted from psi to MPa.")
    df["Start"] = mine._parse_time(df["Start"], warnings)
    df["End"] = mine._parse_time(df["End"], warnings)
    n0 = len(df)
    df = df.dropna(subset=["Stage", "Start", "End", "Volume_m3"])
    df = df[(df["End"] > df["Start"]) & (df["Volume_m3"] > 0)]
    if len(df) < n0:
        warnings.append(f"Dropped {n0 - len(df)} stage rows with invalid times or volumes.")
    if len(df) == 0:
        return None, "No valid stages remain after validation.", warnings
    df["Stage"] = df["Stage"].astype(str).str.strip()
    df = df.sort_values("Start").reset_index(drop=True)
    if (df["Start"].to_numpy()[1:] < df["End"].to_numpy()[:-1]).any():
        warnings.append("Some stage pumping windows overlap; the later stage takes precedence.")
    cols = ["Stage", "Start", "End", "Volume_m3"] + (["Max_Pressure_MPa"] if "Max_Pressure_MPa" in df else [])
    return df[cols], None, warnings


def load_production(file):
    """Production history: Date and Rate_boed (average BOE per day)."""
    warnings = []
    try:
        df = mine._read(file)
    except Exception as exc:
        return None, f"Could not read the production file: {exc}", warnings
    df = mine._normalise(df, _PROD_ALIASES)
    missing = [c for c in ["Date", "Rate_boed"] if c not in df.columns]
    if missing:
        return None, f"Production file missing columns: {missing}. Found: {list(df.columns)}", warnings
    df["Date"] = mine._parse_time(df["Date"], warnings)
    df["Rate_boed"] = pd.to_numeric(df["Rate_boed"], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["Date", "Rate_boed"])
    df = df[df["Rate_boed"] > 0]
    if len(df) < n0:
        warnings.append(f"Dropped {n0 - len(df)} rows with invalid dates or non-positive rates.")
    if len(df) < 6:
        return None, "Need at least 6 valid production points.", warnings
    return df.sort_values("Date")[["Date", "Rate_boed"]].reset_index(drop=True), None, warnings


# END OF FILE: geoflux_io.py