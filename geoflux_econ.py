"""GeoFlux economics: multi-stream monthly discounted cash flow.

Oil, gas and optional NGL streams, royalty, production tax, income tax with
straight-line depreciation, abandonment cost, payback, IRR and breakeven price.
Simplified on purpose (no tax-loss carry-forward, no fiscal-regime specifics):
set the inputs from your own fiscal terms. Any consistent currency.
"""
import numpy as np
from scipy.optimize import brentq

import geoflux_dca as dca

DAYS_PER_MONTH = 30.4375
MCF_PER_BOE = 6.0          # standard gas-to-BOE conversion factor (energy equivalence)

DEFAULT_DECK = {
    "oil_price": 65.0, "gas_price": 3.0, "oil_diff": -3.0, "gas_diff": -0.3,
    "gas_share": 0.30, "ngl_yield": 0.0, "ngl_price_pct": 35.0, "gas_shrink": 0.0,
    "royalty_pct": 12.5, "prod_tax_pct": 5.0, "var_opex": 8.0, "fixed_opex_month": 15000.0,
    "capex": 10e6, "abandon_cost": 150000.0, "tax_pct": 27.0, "dep_years": 7.0,
    "disc_rate_pct": 10.0, "price_mult": 1.0,
}


def cash_flow(q_bpd, deck, with_irr=True):
    """Monthly cash flow for monthly mid-point rates q_bpd (BOE/d, blended).
    gas_share is the gas fraction of BOE volume (6 Mcf = 1 BOE). Production
    stops at the first month with negative operating cash flow."""
    d = {**DEFAULT_DECK, **deck}
    q = np.asarray(q_bpd, dtype=float)
    vol = q * DAYS_PER_MONTH
    gas_mcf = vol * d["gas_share"] * MCF_PER_BOE
    oil_bbl = vol * (1.0 - d["gas_share"])
    ngl_bbl = gas_mcf / 1000.0 * d["ngl_yield"]
    m = d["price_mult"]
    revenue = (oil_bbl * (m * d["oil_price"] + d["oil_diff"])
               + gas_mcf * (1.0 - d["gas_shrink"]) * (m * d["gas_price"] + d["gas_diff"])
               + ngl_bbl * d["ngl_price_pct"] / 100.0 * m * d["oil_price"])
    royalty = revenue * d["royalty_pct"] / 100.0
    prod_tax = (revenue - royalty) * d["prod_tax_pct"] / 100.0
    opcf = revenue - royalty - prod_tax - vol * d["var_opex"] - d["fixed_opex_month"]
    bad = np.flatnonzero(opcf < 0)
    n = int(bad[0]) if bad.size else len(q)
    vol, revenue, opcf = vol[:n], revenue[:n], opcf[:n]
    months_dep = int(round(d["dep_years"] * 12))
    dep = np.where(np.arange(n) < months_dep, d["capex"] / max(months_dep, 1), 0.0)
    cf = opcf - np.maximum(opcf - dep, 0.0) * d["tax_pct"] / 100.0
    if n > 0:
        cf = cf.copy()
        cf[-1] -= d["abandon_cost"]
    r_m = (1.0 + d["disc_rate_pct"] / 100.0) ** (1.0 / 12.0) - 1.0
    t = np.arange(1, n + 1)
    npv = float(-d["capex"] + (cf * (1.0 + r_m) ** (-t)).sum())
    pb = np.flatnonzero(np.cumsum(cf) >= d["capex"])
    irr_m = dca._irr_monthly(cf, d["capex"]) if with_irr else None
    return {"eur_boe": float(vol.sum()), "npv": npv, "life_months": n,
            "revenue_total": float(revenue.sum()), "opcf_total": float(opcf.sum()), "cf": cf,
            "payback_months": int(pb[0]) + 1 if pb.size else None,
            "irr_annual": (1.0 + irr_m) ** 12 - 1.0 if irr_m is not None else None}


def breakeven_multiplier(q_bpd, deck, hi=20.0):
    """Price multiplier (applied to oil and gas prices together) at which NPV = 0.
    None if even `hi` times the deck is not enough."""
    def f(mult):
        return cash_flow(q_bpd, {**deck, "price_mult": mult}, with_irr=False)["npv"]
    if f(hi) < 0:
        return None
    if f(1e-6) >= 0:
        return 0.0
    return float(brentq(f, 1e-6, hi))


# END OF FILE: geoflux_econ.py