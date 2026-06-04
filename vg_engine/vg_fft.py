"""
vg_fft.py
---------
Variance-Gamma option pricing via the characteristic function and the
Carr-Madan FFT method.

Why this replaces vg.calculateCallOptPrice:
  * The old pricer does scipy.quad over the VG density with HARDCODED integration
    bounds [-0.99, 2], which truncates tails and mis-prices deep ITM/OTM and
    long maturities.
  * VG has a closed-form characteristic function. Carr-Madan prices the ENTIRE
    strike chain in one FFT -> fast and numerically stable, with no ad-hoc bounds.

Parametrisation (Madan-Carr-Chang 1998), risk-neutral:
    S_T = S_0 * exp( (r - q + omega) * T + X_T )
    X_T ~ VG(sigma, theta, nu)            (a Levy process)
    omega = (1/nu) * ln(1 - theta*nu - 0.5*sigma^2*nu)   (martingale correction)

Characteristic function of ln(S_T):
    Phi(u) = exp( i*u*(ln S0 + (r - q + omega)*T) )
             * (1 - i*u*theta*nu + 0.5*sigma^2*nu*u^2) ^ (-T/nu)
"""

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# Characteristic function
# --------------------------------------------------------------------------- #
def vg_omega(sigma: float, theta: float, nu: float) -> float:
    """Martingale (mean-correcting) drift so E[S_T] = S_0 e^{(r-q)T}."""
    arg = 1.0 - theta * nu - 0.5 * sigma * sigma * nu
    if arg <= 0:
        # parameters outside the admissible region -> push caller away
        return np.nan
    return np.log(arg) / nu


def vg_char_fn(u, S0, T, r, q, sigma, theta, nu):
    """Characteristic function of ln(S_T) under the risk-neutral VG measure."""
    omega = vg_omega(sigma, theta, nu)
    drift = np.log(S0) + (r - q + omega) * T
    i = 1j
    base = 1.0 - i * u * theta * nu + 0.5 * sigma * sigma * nu * (u ** 2)
    return np.exp(i * u * drift) * base ** (-T / nu)


# --------------------------------------------------------------------------- #
# Carr-Madan FFT pricer (whole strike grid at once)
# --------------------------------------------------------------------------- #
def carr_madan_call_grid(S0, T, r, q, sigma, theta, nu,
                         alpha=1.5, N=16384, eta=0.15):
    """
    Price European calls across a log-strike grid via Carr-Madan FFT.

    Returns
    -------
    strikes : np.ndarray   (the grid of strikes)
    prices  : np.ndarray   (call prices at those strikes)
    """
    lambda_ = 2 * np.pi / (N * eta)          # log-strike spacing
    b = N * lambda_ / 2.0                     # half-width of log-strike range
    u = np.arange(N) * eta                    # integration grid (frequency)
    ku = -b + lambda_ * np.arange(N)          # log-strike grid

    # damped characteristic-function transform
    i = 1j
    cf = vg_char_fn(u - (alpha + 1) * i, S0, T, r, q, sigma, theta, nu)
    denom = (alpha ** 2 + alpha - u ** 2) + i * (2 * alpha + 1) * u
    psi = np.exp(-r * T) * cf / denom

    # Simpson weights for accuracy
    simpson = (3 + (-1) ** np.arange(1, N + 1) - np.concatenate(([1], np.zeros(N - 1)))) / 3.0
    integrand = np.exp(i * b * u) * psi * eta * simpson

    fft_vals = np.fft.fft(integrand).real
    call_prices = np.exp(-alpha * ku) / np.pi * fft_vals
    strikes = np.exp(ku)
    return strikes, call_prices


def vg_call_price(K, S0, T, r, q, sigma, theta, nu, **fft_kw):
    """Single (or vector) strike European CALL price via FFT + cubic interpolation."""
    from scipy.interpolate import interp1d
    strikes, prices = carr_madan_call_grid(S0, T, r, q, sigma, theta, nu, **fft_kw)

    # Restrict to a sensible band around the spot (FFT edges are noisy) and
    # interpolate the price curve cubically in log-strike.
    logK_grid = np.log(strikes)
    band = (logK_grid > np.log(S0) - 4) & (logK_grid < np.log(S0) + 4)
    lg, pr = logK_grid[band], prices[band]
    order = np.argsort(lg)
    f = interp1d(lg[order], pr[order], kind="cubic",
                 bounds_error=False, fill_value="extrapolate")

    out = f(np.log(np.atleast_1d(K)))
    # prices can't be negative (tiny numerical undershoot near deep OTM)
    out = np.maximum(out, 0.0)
    return out if np.ndim(K) else float(out[0])


def vg_put_price(K, S0, T, r, q, sigma, theta, nu, **fft_kw):
    """European PUT via put-call parity:  P = C - S0 e^{-qT} + K e^{-rT}."""
    call = vg_call_price(K, S0, T, r, q, sigma, theta, nu, **fft_kw)
    return call - S0 * np.exp(-q * T) + np.atleast_1d(K) * np.exp(-r * T) \
        if np.ndim(K) else \
        call - S0 * np.exp(-q * T) + K * np.exp(-r * T)


# --------------------------------------------------------------------------- #
# Independent verification: direct Gil-Pelaez / quad integration (slow, exact)
# --------------------------------------------------------------------------- #
def vg_call_price_direct(K, S0, T, r, q, sigma, theta, nu, alpha=1.5):
    """
    Same Carr-Madan integrand but integrated with scipy.quad for a single strike.
    Used only to verify the FFT implementation (should match to ~1e-4).
    """
    from scipy.integrate import quad
    i = 1j
    k = np.log(K)

    def integrand(u):
        cf = vg_char_fn(u - (alpha + 1) * i, S0, T, r, q, sigma, theta, nu)
        denom = (alpha ** 2 + alpha - u ** 2) + i * (2 * alpha + 1) * u
        psi = np.exp(-r * T) * cf / denom
        return (np.exp(-i * u * k) * psi).real

    integral, _ = quad(integrand, 0, 200, limit=400)
    return np.exp(-alpha * k) / np.pi * integral
