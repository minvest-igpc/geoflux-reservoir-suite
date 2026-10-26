"""GeoFlux oil & gas: synthetic frac pad and production history for testing."""
import numpy as np
import pandas as pd

import geoflux_dca as dca


def make_synthetic_frac(seed=11, n_stages=12, b_true=1.0, mc_true=-2.05,
                        sigma_base=-2.6, sigma_hot=-1.8, hot_stage=9):
    """Synthetic horizontal well, 12 stages, one fault 80 m off the well near
    stage 9, which is made more seismogenic. Returns (catalogue, stages, well,
    fault, truth). Z is depth below surface (positive down)."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-06-01 06:00")
    length = 1500.0
    well = pd.DataFrame({"Name": "Well A", "X": [0.0, length], "Y": [0.0, 0.0], "Z": [2500.0, 2500.0]})
    fault = pd.DataFrame({"Name": "Fault F1", "X": [1100.0, 1100.0], "Y": [80.0, 80.0], "Z": [2300.0, 2700.0]})
    seg = length / n_stages
    stage_rows, ev = [], []
    for k in range(n_stages):
        start = t0 + pd.Timedelta(hours=4 * k)
        end = start + pd.Timedelta(hours=2)
        vol = float(rng.uniform(450, 750))
        label = str(k + 1)
        stage_rows.append((label, start, end, vol, float(rng.uniform(55, 75))))
        hot = (k + 1) == hot_stage
        sigma = sigma_hot if hot else sigma_base
        n = rng.poisson(vol * 10.0 ** (sigma - b_true * mc_true))
        cx = (k + 0.5) * seg
        err = rng.uniform(10, 25, n)
        x = rng.normal(cx, 35, n)
        y = rng.normal(0, 40, n)
        if hot:
            toward = rng.random(n) < 0.6
            y = np.where(toward, rng.normal(60, 25, n), y)
        z = rng.normal(2500, 25, n)
        in_pump = rng.random(n) < 0.85
        tt = np.where(in_pump, rng.uniform(0, 7200, n), 7200 + rng.uniform(0, 3000, n))
        mags = np.round(mc_true + rng.exponential(1.0 / (b_true * np.log(10)), n), 1)
        for i in range(n):
            ev.append((start + pd.Timedelta(seconds=float(tt[i])),
                       x[i] + rng.normal(0, err[i] / 2), y[i] + rng.normal(0, err[i] / 2),
                       z[i] + rng.normal(0, err[i] / 2), mags[i], err[i], label))
        if hot:   # one larger event near the fault
            ev.append((start + pd.Timedelta(minutes=70), cx, 70.0, 2480.0, 1.5, 15.0, label))
    cat = pd.DataFrame(ev, columns=["Time", "X", "Y", "Z", "Magnitude", "Loc_Error_m", "truth_stage"])
    cat = cat.sort_values("Time").reset_index(drop=True)
    truth_stage = cat.pop("truth_stage").to_numpy()
    stages = pd.DataFrame(stage_rows, columns=["Stage", "Start", "End", "Volume_m3", "Max_Pressure_MPa"])
    return cat, stages, well, fault, {"sigma_base": sigma_base, "sigma_hot": sigma_hot,
                                      "hot_stage": str(hot_stage), "b": b_true, "stage": truth_stage}


def make_synthetic_production(seed=5, months=18, qi=1100.0, di=2.4, b=1.0,
                              noise=0.08, start="2026-08-01"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=months, freq="MS")
    t = dca.production_time_axis(dates)
    q = dca.arps_rate(t, qi, di, b) * np.exp(rng.normal(0, noise, months))
    return pd.DataFrame({"Date": dates, "Rate_boed": q})


# END OF FILE: geoflux_demo.py