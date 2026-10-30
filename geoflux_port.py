"""GeoFlux portfolio of planned wells: every planned well is an independent
draw from its analog population; correlation between wells is then imposed
explicitly (see geoflux_agg), never assumed away."""
import numpy as np
import pandas as pd

import geoflux_agg as agg
import geoflux_typecurve as tc


def build_samples(plan, fits, meta, n=2000, beta=1.0, seed=0):
    """plan: rows with Group, Formation ('All' for any), Lateral_m, Wells.
    Returns (names, groups, X) where X is n x total_wells of EUR samples (BOE),
    plus a list of problems (groups without enough analogs)."""
    lateral = dict(zip(meta["Well"], meta["Lateral_m"])) if "Lateral_m" in meta else None
    form = dict(zip(meta["Well"], meta["Formation"])) if "Formation" in meta else {}
    names, groups, cols, problems = [], [], [], []
    for i, row in enumerate(plan):
        analogs = {w: f for w, f in fits.items() if row["Formation"] in ("All", form.get(w))}
        if len(analogs) < 5:
            problems.append(f"{row['Group']}: only {len(analogs)} analog wells (need at least 5); skipped.")
            continue
        for j in range(int(row["Wells"])):
            s = tc.analog_samples(analogs, lateral, n, float(row["Lateral_m"]), beta,
                                  seed=seed * 1000 + i * 100 + j, with_rates=False)
            names.append(f"{row['Group']} #{j + 1}")
            groups.append(row["Group"])
            cols.append(s["eur"])
    return names, groups, (np.column_stack(cols) if cols else np.empty((n, 0))), problems


def group_tables(X, groups, names, rank_corr, seed=0):
    """Per-group totals from the correlated samples, plus the arithmetic sums of
    the individual wells' percentiles for comparison."""
    Y = agg.correlated_samples(X, rank_corr, seed)
    wells = agg.well_table({nm: X[:, k] for k, nm in enumerate(names)})
    wells["Group"] = groups
    rows = []
    for g in dict.fromkeys(groups):
        cols = [k for k, gg in enumerate(groups) if gg == g]
        p = agg.percentiles(Y[:, cols].sum(axis=1))
        a = wells[wells["Group"] == g][["P90", "P50", "P10", "Mean"]].sum()
        rows.append({"Group": g, "Wells": len(cols), "P90": p["P90"], "P50": p["P50"], "P10": p["P10"],
                     "Mean": p["Mean"], "Sum_of_P90s": a["P90"], "Sum_of_P10s": a["P10"]})
    return pd.DataFrame(rows), Y


def correlation_curve(X, rhos=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), seed=0):
    return pd.DataFrame([{"Correlation": r, **{k: v for k, v in agg.percentiles(
        agg.correlated_samples(X, r, seed).sum(axis=1)).items()}} for r in rhos])


# END OF FILE: geoflux_port.py