"""
config.py
---------
Central configuration for the VG Greeks engine and hedging simulator.
All tunable parameters live here -- no magic numbers in the modules.
"""

from __future__ import annotations
from typing import TypedDict


class BumpConfig(TypedDict):
    """Finite-difference bump sizes (all relative unless noted)."""
    spot_rel: float        # relative spot bump for delta/vanna
    vol_rel: float         # relative sigma bump for vega/volga
    vol_abs_floor: float   # absolute floor on the sigma bump (1 bp)
    time_days: float       # time bump in *trading days* for theta
    time_max_frac: float   # cap: time bump <= this fraction of remaining T


class FFTConfig(TypedDict):
    """Carr-Madan FFT settings (passed through to vg_fft)."""
    alpha: float
    N: int
    eta: float


class HedgeConfig(TypedDict):
    """Delta-hedging simulation settings."""
    n_paths: int           # paths per engine (GBM and VG)
    horizon_days: int      # option life in trading days
    trading_days: int      # trading days per year
    tc_bps: float          # transaction cost in bps of traded notional
    seed: int              # RNG seed for reproducibility
    spot_grid_n: int       # spot-grid resolution for the delta lookup table
    spot_grid_width: float # grid half-width in units of total vol (sigma*sqrt(T))


class MarketConfig(TypedDict):
    """Default market/measure parameters."""
    r: float               # risk-free rate (flat)
    q: float               # dividend yield


BUMPS: BumpConfig = {
    "spot_rel": 0.001,        # 0.1% of spot
    "vol_rel": 0.01,          # 1% of sigma ...
    "vol_abs_floor": 0.0001,  # ... but never below 1 bp absolute
    "time_days": 1.0,         # 1 trading day
    "time_max_frac": 0.10,    # never more than 10% of remaining life
}

FFT: FFTConfig = {
    "alpha": 1.5,
    "N": 16384,
    "eta": 0.15,
}

HEDGE: HedgeConfig = {
    "n_paths": 1000,
    "horizon_days": 30,
    "trading_days": 252,
    "tc_bps": 2.0,
    "seed": 42,
    "spot_grid_n": 121,
    "spot_grid_width": 6.0,   # +/- 6 total standard deviations
}

MARKET: MarketConfig = {
    "r": 0.066,
    "q": 0.012,
}

# Default VG parameters: the mean calibrated values across 27 independent
# NIFTY trading days (see overfit_check.py output) -- a realistic mid-regime.
VG_PARAMS_DEFAULT: dict[str, float] = {
    "sigma": 0.133,
    "theta": -0.150,
    "nu": 0.114,
}
