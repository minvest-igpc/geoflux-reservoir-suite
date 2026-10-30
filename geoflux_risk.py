"""GeoFlux economic risk: input ranges, Monte Carlo combined with decline
uncertainty, tornado chart and price sensitivity.

Each uncertain input is a triangular distribution (low, base, high). Oil and gas
prices can be correlated. Decline uncertainty comes from the simulated rate
samples, paired one-to-one with the economic draws (independent of them).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

import geoflux_agg as agg
import geoflux_econ as econ

DEFAULT_SPREAD = {"oil_price": 0.30, "gas_price": 0.40, "capex": 0.15,
                  "var_opex": 0.20, "fixed_opex_month": 0.20}
LABELS = {"oil_price": "Oil price", "gas_price": "Gas price", "capex": "Capex",
          "var_opex": "Variable opex", "fixed_opex_month": "Fixed opex"}


def default_spec(deck, spreads=None):
    """{input: (low, base, high)} from the deck values and +/- fractional spreads."""
    sp = {**DEFAULT_SPREAD, **(spreads or {})}
    d = {**econ.DEFAULT_DECK, **deck}
    return {k: (d[k] * (1 - s), d[k], d[k] * (1 + s)) for k, s in sp.items()}


def check_spec(spec):
    for k, (lo, base, hi) in spec.items():
        if not lo <= base <= hi:
            raise ValueError(f"{k}: need low <= base <= high (got {lo}, {base}, {hi})")


def _tri_ppf(u, lo, mode, hi):
    if hi <= lo:
        return np.full_like(u, lo, dtype=float)
    c = (mode - lo) / (hi - lo)
    return np.where(u < c, lo + np.sqrt(u * (hi - lo) * (mode - lo)),
                    hi - np.sqrt((1.0 - u) * (hi - lo) * (hi - mode)))


def sample_inputs(spec, n, seed=0, price_corr=0.5):
    """Triangular draws for each input; oil and gas price linked by a Gaussian
    copula with correlation price_corr."""
    check_spec(spec)
    rng = np.random.default_rng(seed)
    keys = list(spec)
    U = rng.random((n, len(keys)))
    if "oil_price" in spec and "gas_price" in spec:
        z1 = rng.standard_normal(n)
        z2 = price_corr * z1 + np.sqrt(max(1.0 - price_corr ** 2, 0.0)) * rng.standard_normal(n)
        U[:, keys.index("oil_price")], U[:, keys.index("gas_price")] = norm.cdf(z1), norm.cdf(z2)
    return {k: _tri_ppf(U[:, i], *spec[k]) for i, k in enumerate(keys)}


def monte_carlo(rates, deck, spec, seed=0, price_corr=0.5):
    """One economic draw per decline sample (rates: n x months). Returns NPV,
    EUR and the input draws, all aligned by sample."""
    n = len(rates)
    draws = sample_inputs(spec, n, seed, price_corr)
    npv, eur = np.empty(n), np.empty(n)
    for i in range(n):
        r = econ.cash_flow(rates[i], {**deck, **{k: float(v[i]) for k, v in draws.items()}}, with_irr=False)
        npv[i], eur[i] = r["npv"], r["eur_boe"]
    return {"npv": npv, "eur": eur, "draws": draws}


def summarise(npv):
    p = agg.percentiles(npv)
    return {**p, "P_positive": float((np.asarray(npv) > 0).mean())}


def pick_case(rates, eur, pct):
    """The simulated rate curve whose EUR is closest to the given percentile of EUR."""
    idx = int(np.argmin(np.abs(np.asarray(eur) - np.percentile(eur, pct))))
    return rates[idx]


def _fmt(v):
    return f"{v / 1e6:,.2f} M" if abs(v) >= 1e6 else f"{v:,.0f}" if abs(v) >= 1e3 else f"{v:,.4g}"


def tornado(rates_base, rates_p90, rates_p10, deck, spec):
    """One-at-a-time sensitivity of NPV: each input at its low and high value
    with everything else at base. 'Decline' uses the simulated curves closest
    to the P90 (low) and P10 (high) EUR."""
    def npv(q, **over):
        return econ.cash_flow(q, {**deck, **over}, with_irr=False)["npv"]
    base = npv(rates_base)
    rows = [{"Driver": LABELS.get(k, k), "Low_setting": _fmt(lo), "High_setting": _fmt(hi),
             "NPV_at_low": npv(rates_base, **{k: lo}), "NPV_at_high": npv(rates_base, **{k: hi})}
            for k, (lo, _b, hi) in spec.items()]
    rows.append({"Driver": "Decline (EUR P90 / P10)", "Low_setting": "P90 case", "High_setting": "P10 case",
                 "NPV_at_low": npv(rates_p90), "NPV_at_high": npv(rates_p10)})
    df = pd.DataFrame(rows)
    df["Swing"] = (df["NPV_at_high"] - df["NPV_at_low"]).abs()
    return base, df.sort_values("Swing", ascending=False).reset_index(drop=True)


def price_curve(cases, deck, mults):
    """NPV against the oil price (gas moves proportionally) for each rate case."""
    rows = [{"Oil_price": m * deck.get("oil_price", econ.DEFAULT_DECK["oil_price"]),
             "NPV": econ.cash_flow(q, {**deck, "price_mult": m}, with_irr=False)["npv"], "Case": name}
            for name, q in cases.items() for m in mults]
    return pd.DataFrame(rows)


# END OF FILE: geoflux_risk.py