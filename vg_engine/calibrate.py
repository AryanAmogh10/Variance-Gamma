"""
calibrate.py
------------
Calibrate Variance-Gamma parameters to a REAL market option chain by minimising
pricing error -- the upgrade that turns the project from "fit a curve to
historical returns" into "an options-pricing model calibrated to the market".

We calibrate (sigma, theta, nu) under the risk-neutral measure to OTM option
prices (the liquid, information-rich wings), subject to the VG admissibility
constraint  1 - theta*nu - 0.5*sigma^2*nu > 0.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

from vg_fft import vg_call_price, vg_put_price
from bsm import bsm_price, implied_vol


def _model_price(K, typ, S0, T, r, q, sigma, theta, nu):
    if typ == "C":
        return vg_call_price(K, S0, T, r, q, sigma, theta, nu)
    return vg_put_price(K, S0, T, r, q, sigma, theta, nu)


def calibrate_vg(S0, T, r, q, strikes, prices, types,
                 x0=(0.15, -0.10, 0.30)):
    """
    Fit (sigma, theta, nu) to market option prices.

    Parameters
    ----------
    strikes, prices, types : arrays (types: 'C'/'P')
    x0                      : initial (sigma, theta, nu)

    Returns
    -------
    dict with params and fit diagnostics.
    """
    strikes = np.asarray(strikes, float)
    prices = np.asarray(prices, float)
    types = np.asarray(types)

    # relative-error residuals (balances cheap OTM vs expensive ITM)
    def residuals(x):
        sigma, theta, nu = x
        # keep optimiser inside the admissible region
        if 1.0 - theta * nu - 0.5 * sigma ** 2 * nu <= 1e-6 or sigma <= 0 or nu <= 0:
            return np.full(len(strikes), 1e3)
        res = []
        for K, mkt, typ in zip(strikes, prices, types):
            model = _model_price(K, typ, S0, T, r, q, sigma, theta, nu)
            res.append((model - mkt) / max(mkt, 1.0))
        return np.array(res)

    sol = least_squares(
        residuals, x0=np.array(x0),
        bounds=([1e-3, -2.0, 1e-3], [2.0, 2.0, 5.0]),
        method="trf", max_nfev=500, xtol=1e-10, ftol=1e-10,
    )
    sigma, theta, nu = sol.x

    # diagnostics
    model_prices = np.array([_model_price(K, t, S0, T, r, q, sigma, theta, nu)
                             for K, t in zip(strikes, types)])
    rmse = float(np.sqrt(np.mean((model_prices - prices) ** 2)))
    mape = float(np.mean(np.abs((model_prices - prices) / np.maximum(prices, 1e-6))) * 100)

    return {
        "sigma": sigma, "theta": theta, "nu": nu,
        "rmse": rmse, "mape": mape,
        "model_prices": model_prices,
        "success": sol.success, "cost": float(sol.cost),
    }


def benchmark_vs_bsm(S0, T, r, q, strikes, prices, types, vg_params):
    """
    Compare calibrated VG against a single-vol (ATM) Black-Scholes model --
    the apples-to-apples test of "does VG capture the skew that flat BSM can't".

    Returns a dict of RMSE/MAPE for each model plus per-strike detail.
    """
    strikes = np.asarray(strikes, float)
    prices = np.asarray(prices, float)
    types = np.asarray(types)
    sigma, theta, nu = vg_params["sigma"], vg_params["theta"], vg_params["nu"]

    # market implied vols
    mkt_iv = np.array([implied_vol(p, K, S0, T, r, q, t)
                       for K, p, t in zip(strikes, prices, types)])

    # ATM implied vol = single BSM vol (the naive benchmark)
    atm_idx = int(np.argmin(np.abs(strikes - S0)))
    atm_vol = mkt_iv[atm_idx] if np.isfinite(mkt_iv[atm_idx]) else np.nanmedian(mkt_iv)

    bsm_prices = np.array([bsm_price(K, S0, T, r, q, atm_vol, t)
                           for K, t in zip(strikes, types)])
    vg_prices = vg_params["model_prices"]

    def stats(model):
        rmse = np.sqrt(np.mean((model - prices) ** 2))
        mape = np.mean(np.abs((model - prices) / np.maximum(prices, 1e-6))) * 100
        return float(rmse), float(mape)

    vg_rmse, vg_mape = stats(vg_prices)
    bsm_rmse, bsm_mape = stats(bsm_prices)

    return {
        "atm_vol": float(atm_vol),
        "vg_rmse": vg_rmse, "vg_mape": vg_mape,
        "bsm_rmse": bsm_rmse, "bsm_mape": bsm_mape,
        "mkt_iv": mkt_iv, "bsm_prices": bsm_prices, "vg_prices": vg_prices,
    }
