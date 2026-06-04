"""
bsm.py
------
Black-Scholes-Merton pricing and implied-volatility inversion -- the benchmark
the VG model is measured against.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bsm_price(K, S0, T, r, q, sigma, option="C"):
    """Black-Scholes European option price (call or put)."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S0 - K, 0.0) if option == "C" else max(K - S0, 0.0)
        return intrinsic
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "C":
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(price, K, S0, T, r, q, option="C"):
    """Invert BSM for implied volatility. Returns np.nan if no arbitrage-free root."""
    if T <= 0 or price <= 0:
        return np.nan
    # no-arbitrage bounds check
    intrinsic = (max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0) if option == "C"
                 else max(K * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0))
    if price < intrinsic - 1e-8:
        return np.nan

    def objective(sig):
        return bsm_price(K, S0, T, r, q, sig, option) - price

    try:
        return brentq(objective, 1e-4, 5.0, maxiter=200, xtol=1e-8)
    except (ValueError, RuntimeError):
        return np.nan
