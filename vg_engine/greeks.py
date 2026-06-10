"""
greeks.py
---------
Greeks for VG-priced options via central finite differences on the
Carr-Madan FFT pricer.

Vectorisation strategy
----------------------
The FFT prices an ENTIRE strike grid in one call, so every Greek needs only a
handful of FFT evaluations per expiry (one per bump), independent of how many
strikes are requested. A full strike x expiry chain costs
O(n_expiries * n_bumps) FFTs.

Sign conventions
----------------
* delta : dV/dS                > 0 for calls, < 0 for puts
* vega  : dV/d(sigma)          > 0 for vanilla options (per unit of vol)
* volga : d2V/d(sigma)^2       typically > 0 away from ATM
* vanna : d2V/(dS d sigma)
* theta : dV/dt (CALENDAR time) = -dV/dT; negative for long vanillas

Identities tested in tests/test_greeks.py
-----------------------------------------
* delta_call - delta_put = exp(-q*T)        (dividend-adjusted parity)
* vega_call == vega_put                     (parity in sigma)
* central-difference consistency under bump refinement
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from vg_fft import vg_call_price, vg_put_price
from config import BUMPS, MARKET, VG_PARAMS_DEFAULT

OptionType = Literal["C", "P"]


# --------------------------------------------------------------------------- #
# Core: price a strike vector once (one FFT call)
# --------------------------------------------------------------------------- #
def _price_vec(K: np.ndarray, typ: OptionType, S0: float, T: float,
               r: float, q: float, sigma: float, theta: float,
               nu: float) -> np.ndarray:
    """Price a vector of strikes with a single FFT pass.

    Parameters
    ----------
    K : np.ndarray
        Strike vector.
    typ : {'C', 'P'}
        Option type.
    S0, T, r, q, sigma, theta, nu : float
        Spot, year-fraction to expiry, rate, dividend yield, VG parameters.

    Returns
    -------
    np.ndarray
        Option prices at each strike.
    """
    f = vg_call_price if typ == "C" else vg_put_price
    return np.atleast_1d(f(np.asarray(K, float), S0, T, r, q, sigma, theta, nu))


def _bumps_for(S0: float, sigma: float, T: float) -> tuple[float, float, float]:
    """Adaptive central-difference bump sizes (dS, dSigma, dT) from config."""
    dS = BUMPS["spot_rel"] * S0
    dSig = max(BUMPS["vol_rel"] * sigma, BUMPS["vol_abs_floor"])
    dT = min(BUMPS["time_days"] / 252.0, BUMPS["time_max_frac"] * T)
    return dS, dSig, dT


# --------------------------------------------------------------------------- #
# Greeks (vectorised over strikes; loop only over bumps)
# --------------------------------------------------------------------------- #
def greeks_chain(K: np.ndarray, typ: OptionType, S0: float, T: float,
                 r: float = MARKET["r"], q: float = MARKET["q"],
                 sigma: float = VG_PARAMS_DEFAULT["sigma"],
                 theta: float = VG_PARAMS_DEFAULT["theta"],
                 nu: float = VG_PARAMS_DEFAULT["nu"]) -> dict[str, np.ndarray]:
    """Compute price, delta, vega, volga, vanna and theta for a strike vector.

    All Greeks use central finite differences with adaptive bump sizes from
    ``config.BUMPS``. Eleven FFT passes price the full strike vector.

    Parameters
    ----------
    K : np.ndarray
        Strikes (any length).
    typ : {'C', 'P'}
        Option type.
    S0 : float
        Spot.
    T : float
        Year fraction to expiry (> 0).
    r, q : float, optional
        Risk-free rate and dividend yield (default: ``config.MARKET``).
    sigma, theta, nu : float, optional
        VG parameters (default: ``config.VG_PARAMS_DEFAULT``).

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``price, delta, vega, volga, vanna, theta`` -- each an array
        aligned with ``K``.
    """
    K = np.asarray(K, float)
    dS, dSig, dT = _bumps_for(S0, sigma, T)

    P = lambda s0, sig, t: _price_vec(K, typ, s0, t, r, q, sig, theta, nu)

    base = P(S0, sigma, T)
    up_S, dn_S = P(S0 + dS, sigma, T), P(S0 - dS, sigma, T)
    up_V, dn_V = P(S0, sigma + dSig, T), P(S0, sigma - dSig, T)

    delta = (up_S - dn_S) / (2 * dS)
    vega = (up_V - dn_V) / (2 * dSig)
    volga = (up_V - 2 * base + dn_V) / (dSig ** 2)

    # vanna: cross second derivative via four corner bumps
    pp = P(S0 + dS, sigma + dSig, T)
    pm = P(S0 + dS, sigma - dSig, T)
    mp = P(S0 - dS, sigma + dSig, T)
    mm = P(S0 - dS, sigma - dSig, T)
    vanna = (pp - pm - mp + mm) / (4 * dS * dSig)

    # theta: dV/dt = -dV/dT (central in T; falls back to backward at boundary)
    if T - dT > 1e-6:
        up_T, dn_T = P(S0, sigma, T + dT), P(S0, sigma, T - dT)
        theta_g = -(up_T - dn_T) / (2 * dT)
    else:
        up_T = P(S0, sigma, T + dT)
        theta_g = -(up_T - base) / dT

    return {"price": base, "delta": delta, "vega": vega,
            "volga": volga, "vanna": vanna, "theta": theta_g}


def greeks_surface(strikes: np.ndarray, tenors: np.ndarray, typ: OptionType,
                   S0: float, **kw) -> dict[str, np.ndarray]:
    """Compute Greeks over a full strike x tenor grid.

    Parameters
    ----------
    strikes : np.ndarray, shape (nK,)
    tenors : np.ndarray, shape (nT,)
        Year fractions, all > 0.
    typ : {'C', 'P'}
    S0 : float
    **kw
        Forwarded to :func:`greeks_chain` (r, q, sigma, theta, nu).

    Returns
    -------
    dict[str, np.ndarray]
        Each value has shape (nT, nK): rows = tenors, cols = strikes.
    """
    rows: dict[str, list[np.ndarray]] = {}
    for T in np.asarray(tenors, float):
        g = greeks_chain(np.asarray(strikes, float), typ, S0, float(T), **kw)
        for name, vals in g.items():
            rows.setdefault(name, []).append(vals)
    return {name: np.vstack(v) for name, v in rows.items()}


# --------------------------------------------------------------------------- #
# Visualiser
# --------------------------------------------------------------------------- #
def plot_greeks_surface(strikes: np.ndarray, tenors: np.ndarray, typ: OptionType,
                        S0: float, out_path: str | None = None,
                        **kw) -> "matplotlib.figure.Figure":
    """Heatmap panel of all Greeks over moneyness x tenor.

    Parameters
    ----------
    strikes, tenors, typ, S0
        As in :func:`greeks_surface`.
    out_path : str, optional
        If given, the figure is saved there (PNG) and closed.
    **kw
        Forwarded to :func:`greeks_surface`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surf = greeks_surface(strikes, tenors, typ, S0, **kw)
    moneyness = np.asarray(strikes, float) / S0
    names = ["price", "delta", "vega", "volga", "vanna", "theta"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, name in zip(axes.flat, names):
        z = surf[name]
        im = ax.pcolormesh(moneyness, tenors, z, shading="auto", cmap="RdBu_r")
        ax.set_title(f"VG {name} ({'call' if typ == 'C' else 'put'})")
        ax.set_xlabel("moneyness K/S0")
        ax.set_ylabel("tenor (years)")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
    return fig
