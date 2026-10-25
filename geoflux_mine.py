"""GeoFlux mine seismology module.

Catalogue loading, rule-based blast discrimination, 3-D distance to workings,
activity-rate flags and statistical hazard screening, plus a synthetic mine
generator for testing before real data is available.

IMPORTANT: blast discrimination here uses timestamps, blast logs and hours of
day. It does NOT analyse waveforms. Seismicity inside a blast window cannot be
separated from the blast signal without waveform features.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

from geoflux_core import (energy_to_magnitude, magnitude_to_energy,
                          estimate_b_value, exceedance_probability)

ALL_CLASSES = ["Seismic", "Post-blast window", "Blast-hour (unconfirmed)",
               "Blast (logged)", "Blast (system-classified)", "Noise (system-classified)"]
# Post-blast seismicity is REAL hazard, so it stays in the analysis by default.
DEFAULT_INCLUDE = ["Seismic", "Post-blast window", "Blast-hour (unconfirmed)"]

_CAT_ALIASES = {
    "time": "Time", "datetime": "Time", "timestamp": "Time", "date_time": "Time",
    "origin_time": "Time", "event_time": "Time",
    "x": "X", "easting": "X", "easting_m": "X",
    "y": "Y", "northing": "Y", "northing_m": "Y",
    "z": "Z", "elevation": "Z", "elevation_m": "Z", "z_m": "Z",
    "magnitude": "Magnitude", "mag": "Magnitude", "local_magnitude": "Magnitude",
    "ml": "Magnitude", "mw": "Magnitude",
    "energy": "Energy_J", "energy_j": "Energy_J", "energy_joules": "Energy_J",
    "moment": "Moment_Nm", "moment_nm": "Moment_Nm", "seismic_moment": "Moment_Nm",
    "event_type": "Event_Type", "type": "Event_Type",
    "loc_error_m": "Loc_Error_m", "location_error_m": "Loc_Error_m", "error_m": "Loc_Error_m",
}
_BLAST_ALIASES = {**{k: v for k, v in _CAT_ALIASES.items() if v in ("Time", "X", "Y", "Z")},
                  "notes": "Notes", "comment": "Notes", "description": "Notes"}
_WORK_ALIASES = {**{k: v for k, v in _CAT_ALIASES.items() if v in ("X", "Y", "Z")},
                 "name": "Name", "working": "Name", "id": "Name", "level": "Name"}


# ------------------------------------------------------------------- loading
def _read(file):
    return pd.read_csv(file, sep=None, engine="python")


def _normalise(df, aliases):
    mapping = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_")
        if key in aliases:
            mapping[c] = aliases[key]
    return df.rename(columns=mapping)


def _parse_time(series, warnings):
    t = pd.to_datetime(series, errors="coerce")
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
        warnings.append("Timezone information was dropped; times are treated as one local clock.")
    return t


def load_catalogue(file):
    """Returns (DataFrame or None, error or None, warnings).
    Required columns: Time, X, Y, Z (mine grid, metres) and Magnitude or Energy_J."""
    warnings = []
    try:
        df = _read(file)
    except Exception as exc:
        return None, f"Could not read the file: {exc}", warnings
    df = _normalise(df, _CAT_ALIASES)
    if df.columns.duplicated().any():
        return None, "Two columns map to the same field; rename or remove one.", warnings
    missing = [c for c in ["Time", "X", "Y", "Z"] if c not in df.columns]
    if missing:
        return None, f"Missing required columns: {missing}. Found: {list(df.columns)}", warnings
    if "Magnitude" not in df.columns and "Energy_J" not in df.columns:
        return None, "Need a Magnitude column or an Energy_J column.", warnings
    df["Time"] = _parse_time(df["Time"], warnings)
    for c in ["X", "Y", "Z", "Magnitude", "Energy_J", "Moment_Nm", "Loc_Error_m"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["Time", "X", "Y", "Z"])
    if len(df) < n0:
        warnings.append(f"Dropped {n0 - len(df)} rows with missing or invalid time/coordinates.")
    if "Magnitude" not in df.columns:
        df["Magnitude"] = energy_to_magnitude(df["Energy_J"])
        warnings.append("Magnitude was derived from Energy_J (Gutenberg-Richter relation). "
                        "Supply the system's magnitude if you have it.")
    n1 = len(df)
    df = df.dropna(subset=["Magnitude"])
    if len(df) < n1:
        warnings.append(f"Dropped {n1 - len(df)} rows with no magnitude.")
    n2 = len(df)
    df = df.drop_duplicates(subset=["Time", "X", "Y", "Z", "Magnitude"])
    if len(df) < n2:
        warnings.append(f"Dropped {n2 - len(df)} exact duplicate events.")
    if len(df) == 0:
        return None, "No valid events remain after validation.", warnings
    keep = ["Time", "X", "Y", "Z", "Magnitude"] + [
        c for c in ["Energy_J", "Moment_Nm", "Event_Type", "Loc_Error_m"] if c in df.columns]
    return df[keep].sort_values("Time").reset_index(drop=True), None, warnings


def load_blast_log(file):
    warnings = []
    try:
        df = _read(file)
    except Exception as exc:
        return None, f"Could not read the blast log: {exc}", warnings
    df = _normalise(df, _BLAST_ALIASES)
    if "Time" not in df.columns:
        return None, f"Blast log needs a Time column. Found: {list(df.columns)}", warnings
    df["Time"] = _parse_time(df["Time"], warnings)
    n0 = len(df)
    df = df.dropna(subset=["Time"])
    if len(df) < n0:
        warnings.append(f"Dropped {n0 - len(df)} blast-log rows with invalid times.")
    if len(df) == 0:
        return None, "No valid blast times in the log.", warnings
    return df.sort_values("Time").reset_index(drop=True), None, warnings


def load_workings(file):
    """Workings as 3-D polylines: columns Name (optional), X, Y, Z. Rows with
    the same Name are joined in file order. A single row is treated as a point."""
    warnings = []
    try:
        df = _read(file)
    except Exception as exc:
        return None, f"Could not read the workings file: {exc}", warnings
    df = _normalise(df, _WORK_ALIASES)
    missing = [c for c in ["X", "Y", "Z"] if c not in df.columns]
    if missing:
        return None, f"Workings file missing columns: {missing}. Found: {list(df.columns)}", warnings
    if "Name" not in df.columns:
        df["Name"] = "Workings"
    for c in ["X", "Y", "Z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["X", "Y", "Z"])
    if len(df) < n0:
        warnings.append(f"Dropped {n0 - len(df)} workings rows with invalid coordinates.")
    if len(df) == 0:
        return None, "No valid workings vertices.", warnings
    df["Name"] = df["Name"].astype(str)
    return df[["Name", "X", "Y", "Z"]].reset_index(drop=True), None, warnings


# ------------------------------------------------------- distance to workings
def _point_segment_distance(P, A, B):
    AB = B - A
    denom = float(AB @ AB)
    if denom == 0.0:
        return np.linalg.norm(P - A, axis=1)
    t = np.clip(((P - A) @ AB) / denom, 0.0, 1.0)
    return np.linalg.norm(P - (A + t[:, None] * AB), axis=1)


def distance_to_workings(df, workings):
    """Nearest 3-D distance (m) from each event to any working polyline.
    Events and workings must share one coordinate system and Z convention."""
    P = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    best = np.full(len(P), np.inf)
    names = np.full(len(P), "", dtype=object)
    for name, g in workings.groupby("Name", sort=False):
        V = g[["X", "Y", "Z"]].to_numpy(dtype=float)
        if len(V) == 1:
            d = np.linalg.norm(P - V[0], axis=1)
        else:
            d = np.min([_point_segment_distance(P, V[i], V[i + 1])
                        for i in range(len(V) - 1)], axis=0)
        closer = d < best
        best[closer] = d[closer]
        names[closer] = name
    return best, names


# ------------------------------------------------------ blast discrimination
def classify_events(df, blast_log=None, blast_window_s=60.0, pre_tol_s=5.0,
                    post_window_min=30.0, blast_hours=(), hour_tol_min=30.0):
    """Rule-based classification (later rules override earlier ones):
      1. Blast-hour (unconfirmed): within hour_tol_min of a typical blast hour.
      2. Post-blast window: after a logged blast, beyond the direct window.
      3. Blast (logged): within blast_window_s after (or pre_tol_s before,
         for clock error) a logged blast.
      4. System classification, if the catalogue has an Event_Type column.
    """
    n = len(df)
    cls = np.full(n, "Seismic", dtype=object)
    if blast_hours:
        t = df["Time"]
        minute_of_day = (t.dt.hour * 60 + t.dt.minute + t.dt.second / 60.0).to_numpy()
        hit = np.zeros(n, dtype=bool)
        for h in blast_hours:
            d = np.abs(minute_of_day - float(h) * 60.0)
            hit |= np.minimum(d, 1440.0 - d) <= hour_tol_min
        cls[hit] = "Blast-hour (unconfirmed)"
    if blast_log is not None and len(blast_log) > 0:
        bt = np.sort(blast_log["Time"].to_numpy("datetime64[ns]"))
        t = df["Time"].to_numpy("datetime64[ns]")
        one_s = np.timedelta64(1, "s")
        i_prev = np.searchsorted(bt, t, side="right") - 1
        dt_after = np.where(i_prev >= 0, (t - bt[np.clip(i_prev, 0, None)]) / one_s, np.inf)
        i_next = np.searchsorted(bt, t, side="left")
        dt_before = np.where(i_next < len(bt),
                             (bt[np.clip(i_next, None, len(bt) - 1)] - t) / one_s, np.inf)
        direct = (dt_after <= blast_window_s) | (dt_before <= pre_tol_s)
        post = (dt_after > blast_window_s) & (dt_after <= post_window_min * 60.0)
        cls[post] = "Post-blast window"
        cls[direct] = "Blast (logged)"
    if "Event_Type" in df.columns:
        s = df["Event_Type"].astype(str).str.lower()
        cls[s.str.contains("blast").to_numpy()] = "Blast (system-classified)"
        cls[s.str.contains("noise").to_numpy()] = "Noise (system-classified)"
    return cls


# ------------------------------------------------------------ rates and flags
def daily_table(df, mc, t_start=None, t_end=None):
    """Daily counts (>= Mc) and summed energy over the full date range."""
    days = pd.date_range((t_start or df["Time"].min()).floor("D"),
                         (t_end or df["Time"].max()).floor("D"), freq="D")
    sel = df[df["Magnitude"] >= mc - 1e-9]
    day = sel["Time"].dt.floor("D")
    out = pd.DataFrame({"Date": days})
    out["Count"] = day.value_counts().reindex(days, fill_value=0).to_numpy()
    if "Energy_J" in sel.columns:
        out["Energy_J"] = sel.groupby(day)["Energy_J"].sum().reindex(days, fill_value=0.0).to_numpy()
    return out


def poisson_rate_flags(daily, baseline_days=30, alpha=0.01):
    """Flag days whose count is improbably high under a Poisson process whose
    rate is the mean of the first baseline_days days. Seismicity is clustered,
    so expect some false alarms; a flag means 'look at this day', not 'hazard'."""
    counts = daily["Count"].to_numpy()
    lam = max(float(counts[:baseline_days].mean()) if len(counts) else 0.0, 0.1)
    p = poisson.sf(counts - 1, lam)           # P(X >= count)
    flag = (p < alpha) & (counts > lam)
    flag[:baseline_days] = False
    out = daily.copy()
    out["Baseline_rate"], out["p_value"], out["Flag"] = lam, p, flag
    return out


# ------------------------------------------------------------ hazard screening
def hazard_screen(df, mc, b_res, recent_days=14, horizon_days=7, thresholds=(0.0, 1.0, 2.0)):
    """Poisson / Gutenberg-Richter exceedance probabilities for the next
    horizon_days, with crude low/high bounds from rate +/- sqrt(N) and b -/+ sigma."""
    sel = df[df["Magnitude"] >= mc - 1e-9]
    t_end = df["Time"].max()
    n = int((sel["Time"] >= t_end - pd.Timedelta(days=recent_days)).sum())
    rate = n / recent_days
    rate_lo, rate_hi = max(n - np.sqrt(n), 0.0) / recent_days, (n + np.sqrt(n)) / recent_days
    b, s = b_res["b"], b_res["sigma"]
    rows = []
    for m_star in thresholds:
        rows.append({
            "Magnitude threshold": m_star,
            "P(>= threshold)": exceedance_probability(rate, b, mc, m_star, horizon_days),
            "Low": exceedance_probability(rate_lo, b + s, mc, m_star, horizon_days),
            "High": exceedance_probability(rate_hi, max(b - s, 0.05), mc, m_star, horizon_days),
        })
    return pd.DataFrame(rows), {"recent_events": n, "rate_per_day": rate,
                                "largest": float(sel["Magnitude"].max()) if len(sel) else np.nan}


# ----------------------------------------------------------- synthetic mine
def make_synthetic_mine(seed=7, days=90, start="2026-07-01", b_true=1.0,
                        mc_true=-1.05, unlogged_frac=0.10):
    """Synthetic mine for testing. Returns (catalogue, blast_log, workings, truth).
    truth labels each event 'background', 'swarm', 'blast' or 'post-blast' and is
    for testing only; the app never shows it as data."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    stope = np.array([[300, 200, -20], [320, 260, -25], [340, 320, -30]], float)
    drive = np.array([[0, 0, 0], [300, 0, 0], [300, 200, -20]], float)
    workings = pd.DataFrame(
        [("Haulage drive", *p) for p in drive] + [("Stope A", *p) for p in stope],
        columns=["Name", "X", "Y", "Z"])

    def mags(n):  # Gutenberg-Richter above mc_true, rounded to 0.1 like real catalogues
        return np.round(mc_true + rng.exponential(1.0 / (b_true * np.log(10.0)), n), 1)

    def near_stope(n, sd=25.0):
        return stope[rng.integers(0, len(stope), n)] + rng.normal(0, sd, (n, 3))

    ev, blasts = [], []

    def add(times, xyz, m, label):
        for ti, p, mi in zip(times, xyz, m):
            ev.append((ti, p[0], p[1], p[2], mi, label))

    for d in range(days):
        day0 = t0 + pd.Timedelta(days=d)
        n = rng.poisson(12)
        near = rng.random(n) < 0.7
        xyz = np.where(near[:, None], near_stope(n),
                       rng.uniform([0, -100, -80], [400, 400, 20], (n, 3)))
        add(day0 + pd.to_timedelta(rng.uniform(0, 86400, n), unit="s"), xyz, mags(n), "background")
        if 55 <= d < 58:  # swarm near the stope
            k = rng.poisson(36)
            add(day0 + pd.to_timedelta(rng.uniform(0, 86400, k), unit="s"),
                near_stope(k, 15.0), mags(k), "swarm")
        for hr in (5.5, 17.5):
            tb = day0 + pd.Timedelta(hours=hr) + pd.Timedelta(seconds=float(rng.normal(0, 120)))
            loc = drive[rng.integers(0, len(drive))] + rng.normal(0, 5, 3)
            if rng.random() > unlogged_frac:
                blasts.append((tb, *loc, "Production blast"))
            add([tb + pd.Timedelta(seconds=float(rng.uniform(1, 15)))], [loc + rng.normal(0, 5, 3)],
                np.round(rng.normal(0.3, 0.3, 1), 1), "blast")
            k = rng.poisson(8)
            delays = 20 + 1780 * rng.random(k) ** 2.5   # Omori-like: concentrated early
            add([tb + pd.Timedelta(seconds=float(s)) for s in delays],
                loc + rng.normal(0, 30, (k, 3)), mags(k), "post-blast")

    cat = pd.DataFrame(ev, columns=["Time", "X", "Y", "Z", "Magnitude", "truth"])
    cat["Energy_J"] = magnitude_to_energy(cat["Magnitude"])
    cat["Loc_Error_m"] = rng.uniform(5, 30, len(cat))
    cat = cat.sort_values("Time").reset_index(drop=True)
    truth = cat.pop("truth").to_numpy()
    blast_log = pd.DataFrame(blasts, columns=["Time", "X", "Y", "Z", "Notes"]).sort_values("Time")
    return cat, blast_log.reset_index(drop=True), workings, truth