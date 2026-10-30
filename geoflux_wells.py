"""GeoFlux wells: multi-well production and metadata loaders, analog selection
and a synthetic basin (two formations, known answers) for testing."""
import numpy as np
import pandas as pd

import geoflux_dca as dca
import geoflux_mine as mine

FT_TO_M = 0.3048

_PROD_ALIASES = {"well": "Well", "well_name": "Well", "uwi": "Well", "api": "Well", "well_id": "Well",
                 "date": "Date", "month": "Date", "period": "Date",
                 "rate_boed": "Rate_boed", "boed": "Rate_boed", "boe_d": "Rate_boed", "rate": "Rate_boed"}
_META_ALIASES = {"well": "Well", "well_name": "Well", "uwi": "Well", "api": "Well", "well_id": "Well",
                 "formation": "Formation", "target": "Formation", "zone": "Formation", "basin": "Basin",
                 "lateral_m": "Lateral_m", "lateral_length_m": "Lateral_m",
                 "lateral_ft": "Lateral_ft", "lateral_length_ft": "Lateral_ft",
                 "completion_year": "Completion_Year", "year": "Completion_Year", "vintage": "Completion_Year",
                 "well_type": "Well_Type", "type": "Well_Type", "stages": "Stages"}


def load_well_production(file):
    """Long-format monthly production: Well, Date, Rate_boed (BOE per day)."""
    warnings = []
    try:
        df = mine._read(file)
    except Exception as exc:
        return None, f"Could not read the production file: {exc}", warnings
    df = mine._normalise(df, _PROD_ALIASES)
    missing = [c for c in ["Well", "Date", "Rate_boed"] if c not in df.columns]
    if missing:
        return None, f"Missing columns: {missing}. Found: {list(df.columns)}", warnings
    df["Date"] = mine._parse_time(df["Date"], warnings)
    df["Rate_boed"] = pd.to_numeric(df["Rate_boed"], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["Well", "Date", "Rate_boed"])
    df = df[df["Rate_boed"] > 0]
    if len(df) < n0:
        warnings.append(f"Dropped {n0 - len(df)} rows with invalid dates or non-positive rates.")
    if df.empty:
        return None, "No valid production rows remain.", warnings
    df["Well"] = df["Well"].astype(str).str.strip()
    return df.sort_values(["Well", "Date"])[["Well", "Date", "Rate_boed"]].reset_index(drop=True), None, warnings


def load_well_meta(file):
    """Well metadata: Well plus any of Formation, Basin, Lateral_m (or Lateral_ft),
    Completion_Year, Well_Type, Stages."""
    warnings = []
    try:
        df = mine._read(file)
    except Exception as exc:
        return None, f"Could not read the metadata file: {exc}", warnings
    df = mine._normalise(df, _META_ALIASES)
    if "Well" not in df.columns:
        return None, f"Metadata needs a Well column. Found: {list(df.columns)}", warnings
    if "Lateral_m" not in df.columns and "Lateral_ft" in df.columns:
        df["Lateral_m"] = pd.to_numeric(df["Lateral_ft"], errors="coerce") * FT_TO_M
        warnings.append("Lateral length converted from feet to metres.")
    for c in ["Lateral_m", "Completion_Year", "Stages"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Well"] = df["Well"].astype(str).str.strip()
    keep = ["Well"] + [c for c in ["Formation", "Basin", "Lateral_m", "Completion_Year", "Well_Type", "Stages"] if c in df.columns]
    return df[keep].drop_duplicates("Well").reset_index(drop=True), None, warnings


def well_series(prod):
    """{well: (years since first record, rate BOE/d)} for every well."""
    return {w: (dca.production_time_axis(g["Date"]), g["Rate_boed"].to_numpy(dtype=float))
            for w, g in prod.groupby("Well", sort=False)}


def select_analogs(meta, formations=None, years=None, lateral=None, well_types=None):
    """Well names passing every filter that the metadata can support."""
    m = meta
    if formations and "Formation" in m:
        m = m[m["Formation"].isin(formations)]
    if years and "Completion_Year" in m:
        m = m[m["Completion_Year"].between(years[0], years[1])]
    if lateral and "Lateral_m" in m:
        m = m[m["Lateral_m"].between(lateral[0], lateral[1])]
    if well_types and "Well_Type" in m:
        m = m[m["Well_Type"].isin(well_types)]
    return list(m["Well"])


def make_synthetic_wells(seed=3, n_wells=36):
    """Synthetic basin: Formation A (about 2/3 of wells, 520 BOE/d per km initial
    rate) and Formation B (380). Wells vary in lateral length, vintage, decline
    and noise. Returns (production, metadata)."""
    rng = np.random.default_rng(seed)
    now = pd.Timestamp("2026-09-01")
    rows, meta = [], []
    for i in range(n_wells):
        form = "Formation B" if i % 3 == 0 else "Formation A"
        year, month = int(rng.integers(2019, 2025)), int(rng.integers(1, 13))
        start = pd.Timestamp(year=year, month=month, day=1)
        lat = float(rng.uniform(1500, 3500))
        qi = {"Formation A": 520.0, "Formation B": 380.0}[form] * np.exp(rng.normal(0, 0.25)) * lat / 1000.0
        di, b = float(np.exp(rng.normal(np.log(1.9), 0.2))), float(np.clip(rng.normal(1.0, 0.12), 0.5, 1.6))
        n_months = (now.year - start.year) * 12 + now.month - start.month
        dates = pd.date_range(start, periods=n_months, freq="MS")
        q = dca.arps_rate(dca.production_time_axis(dates), qi, di, b) * np.exp(rng.normal(0, 0.08, n_months))
        name = f"W{i + 1:03d}"
        rows.append(pd.DataFrame({"Well": name, "Date": dates, "Rate_boed": q}))
        meta.append({"Well": name, "Formation": form, "Basin": "Demo Basin", "Lateral_m": lat,
                     "Completion_Year": year, "Well_Type": "MFHW"})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(meta)


# END OF FILE: geoflux_wells.py