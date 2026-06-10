"""
hedging.py
----------
Daily delta-hedging P&L simulator: VG delta vs BSM delta as the hedge ratio,
on synthetic GBM and VG underlying paths.

Setup (per path)
----------------
* Sell one ATM European call at the model price, life = HEDGE['horizon_days'].
* Each day: recompute the hedge delta, rebalance the share position, pay
  transaction costs of HEDGE['tc_bps'] bps of traded notional.
* Cash accrues at the risk-free rate; at expiry the option is settled.
* Hedging error = terminal portfolio value (perfect replication => 0).

Performance design
------------------
Deltas are NOT recomputed by FFT per path-step (that would be ~120k FFT
calls). Each hedge model's delta is precomputed once on a spot x time lattice
(one FFT batch per time step), then bilinearly interpolated along the paths.
Grid spans +/- HEDGE['spot_grid_width'] total standard deviations, so paths
essentially never leave it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from vg_fft import vg_call_price, vg_omega
from bsm import bsm_price, implied_vol
from greeks import greeks_chain
from config import HEDGE, MARKET, VG_PARAMS_DEFAULT


# --------------------------------------------------------------------------- #
# Path simulation
# --------------------------------------------------------------------------- #
def simulate_gbm_paths(S0: float, vol: float, T: float, n_steps: int,
                       n_paths: int, r: float, q: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Simulate risk-neutral GBM paths.

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps + 1)
        Spot paths including S0 at column 0.
    """
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_inc = (r - q - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z
    log_paths = np.cumsum(log_inc, axis=1)
    return S0 * np.hstack([np.ones((n_paths, 1)),
                           np.exp(log_paths)])


def simulate_vg_paths(S0: float, sigma: float, theta: float, nu: float,
                      T: float, n_steps: int, n_paths: int, r: float,
                      q: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate risk-neutral VG paths via gamma time subordination.

    Increments: dX = theta*dG + sigma*sqrt(dG)*Z,  dG ~ Gamma(dt/nu, nu),
    drift = (r - q + omega) dt with omega the VG martingale correction --
    identical measure to the FFT pricer.

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps + 1)
    """
    dt = T / n_steps
    omega = vg_omega(sigma, theta, nu)
    g = rng.gamma(shape=dt / nu, scale=nu, size=(n_paths, n_steps))
    z = rng.standard_normal((n_paths, n_steps))
    log_inc = (r - q + omega) * dt + theta * g + sigma * np.sqrt(g) * z
    log_paths = np.cumsum(log_inc, axis=1)
    return S0 * np.hstack([np.ones((n_paths, 1)),
                           np.exp(log_paths)])


# --------------------------------------------------------------------------- #
# Delta lookup tables (spot x time lattice)
# --------------------------------------------------------------------------- #
@dataclass
class DeltaTable:
    """Bilinear-interpolated delta surface over (time-step, spot)."""
    spot_grid: np.ndarray            # (nS,)
    deltas: np.ndarray               # (n_steps, nS); row i = step i (T_i left)

    def lookup(self, step: int, spots: np.ndarray) -> np.ndarray:
        """Delta at time-step `step` for an array of spots (clipped to grid)."""
        s = np.clip(spots, self.spot_grid[0], self.spot_grid[-1])
        return np.interp(s, self.spot_grid, self.deltas[step])


def _spot_lattice(S0: float, total_vol: float) -> np.ndarray:
    """Log-spaced spot grid spanning +/- grid_width total stdevs."""
    w = HEDGE["spot_grid_width"] * total_vol
    return S0 * np.exp(np.linspace(-w, w, HEDGE["spot_grid_n"]))


def build_vg_delta_table(K: float, S0: float, T: float, n_steps: int,
                         r: float, q: float, sigma: float, theta: float,
                         nu: float) -> DeltaTable:
    """Precompute the VG call delta on a spot x time lattice.

    One :func:`greeks_chain`-style FD evaluation per (step, spot) pair, but
    batched: per time step we price the whole spot lattice with two FFT passes
    (spot up / spot down) by exploiting FFT homogeneity is NOT valid for
    spot bumps, so we simply loop the lattice -- still only
    n_steps * nS * 2 one-dimensional FFT interpolations, done once.
    """
    total_vol = sigma * np.sqrt(T) * 1.5 + 0.05
    grid = _spot_lattice(S0, total_vol)
    dt = T / n_steps
    h_rel = 0.001

    deltas = np.empty((n_steps, len(grid)))
    for i in range(n_steps):
        tau = T - i * dt                       # time remaining at step i
        for j, s in enumerate(grid):
            h = h_rel * s
            up = vg_call_price(K, s + h, tau, r, q, sigma, theta, nu)
            dn = vg_call_price(K, s - h, tau, r, q, sigma, theta, nu)
            deltas[i, j] = (up - dn) / (2 * h)
    return DeltaTable(grid, deltas)


def build_bsm_delta_table(K: float, S0: float, T: float, n_steps: int,
                          r: float, q: float, vol: float) -> DeltaTable:
    """Closed-form BSM call delta on the same lattice (vectorised exactly)."""
    total_vol = vol * np.sqrt(T) * 1.5 + 0.05
    grid = _spot_lattice(S0, total_vol)
    dt = T / n_steps
    deltas = np.empty((n_steps, len(grid)))
    for i in range(n_steps):
        tau = max(T - i * dt, 1e-8)
        d1 = (np.log(grid / K) + (r - q + 0.5 * vol ** 2) * tau) / (vol * np.sqrt(tau))
        deltas[i] = np.exp(-q * tau) * norm.cdf(d1)
    return DeltaTable(grid, deltas)


# --------------------------------------------------------------------------- #
# Hedging engine
# --------------------------------------------------------------------------- #
@dataclass
class HedgeResult:
    """Per-path results of a delta-hedging simulation."""
    label: str
    hedge_error: np.ndarray          # terminal portfolio value per path
    pnl_paths: np.ndarray            # (n_paths, n_steps+1) cumulative P&L
    delta_paths: np.ndarray          # (n_paths, n_steps) hedge ratio held
    costs: np.ndarray                # total transaction cost per path
    premium: float

    @property
    def mae(self) -> float:
        """Mean absolute hedging error."""
        return float(np.mean(np.abs(self.hedge_error)))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.hedge_error ** 2)))

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe of per-path total P&L over the episode."""
        mu, sd = self.hedge_error.mean(), self.hedge_error.std()
        if sd == 0:
            return 0.0
        episodes_per_year = HEDGE["trading_days"] / HEDGE["horizon_days"]
        return float(mu / sd * np.sqrt(episodes_per_year))


def run_hedge(paths: np.ndarray, table: DeltaTable, K: float, premium: float,
              T: float, r: float, label: str,
              tc_bps: float = HEDGE["tc_bps"],
              q: float = MARKET["q"]) -> HedgeResult:
    """Run a daily short-call delta hedge along simulated paths.

    Parameters
    ----------
    paths : np.ndarray, shape (n_paths, n_steps + 1)
        Underlying paths (col 0 = S0).
    table : DeltaTable
        Precomputed hedge-delta lattice for this model.
    K, premium, T, r : float
        Strike, option premium received, option life (years), risk-free rate.
    label : str
        Name for reporting ('VG delta', 'BSM delta').
    tc_bps : float
        One-way transaction cost in bps of traded notional.
    q : float
        Continuous dividend yield credited on the share position. The pricing
        measure drifts the stock at (r - q) because holders receive q; the
        hedge portfolio must therefore earn q on its stock leg or the
        replication leaks ~mean(delta)*S*q*T per episode.

    Returns
    -------
    HedgeResult
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps
    tc = tc_bps * 1e-4
    growth = np.exp(r * dt)

    delta_paths = np.empty((n_paths, n_steps))
    pnl_paths = np.empty((n_paths, n_cols))
    costs = np.zeros(n_paths)

    # t = 0: sell option (receive premium), buy delta0 shares
    d0 = table.lookup(0, paths[:, 0])
    cost0 = tc * np.abs(d0) * paths[:, 0]
    cash = premium - d0 * paths[:, 0] - cost0
    shares = d0.copy()
    costs += cost0
    delta_paths[:, 0] = d0
    pnl_paths[:, 0] = 0.0

    div_growth = np.exp(q * dt) - 1.0

    for i in range(1, n_steps):
        s = paths[:, i]
        cash *= growth
        cash += shares * paths[:, i - 1] * div_growth   # dividend income on stock leg
        d_new = table.lookup(i, s)
        trade = d_new - shares
        cost_i = tc * np.abs(trade) * s
        cash -= trade * s + cost_i
        costs += cost_i
        shares = d_new
        delta_paths[:, i] = d_new
        # mark-to-market: cash + stock - option intrinsic proxy is noisy; for
        # the P&L track we report cash + stock - (premium accreted), i.e.
        # portfolio value relative to t=0
        pnl_paths[:, i] = cash + shares * s - premium * np.exp(r * i * dt)

    # expiry: liquidate shares, settle option payoff
    sT = paths[:, -1]
    cash *= growth
    cash += shares * paths[:, -2] * div_growth          # final day's dividend
    cash += shares * sT - tc * np.abs(shares) * sT
    costs += tc * np.abs(shares) * sT
    payoff = np.maximum(sT - K, 0.0)
    terminal = cash - payoff
    pnl_paths[:, -1] = terminal

    return HedgeResult(label=label, hedge_error=terminal, pnl_paths=pnl_paths,
                       delta_paths=delta_paths, costs=costs, premium=premium)


# --------------------------------------------------------------------------- #
# Full experiment
# --------------------------------------------------------------------------- #
def run_experiment(S0: float = 23700.0,
                   vg_params: dict[str, float] | None = None,
                   market: dict[str, float] | None = None,
                   n_paths: int = HEDGE["n_paths"],
                   horizon_days: int = HEDGE["horizon_days"],
                   seed: int = HEDGE["seed"]) -> dict[str, HedgeResult]:
    """Run the full 2x2 hedging experiment: {GBM, VG} paths x {VG, BSM} delta.

    The short option is the ATM call (K = S0). The BSM hedger uses the
    VG-implied ATM vol (invert the VG ATM price through BSM), i.e. the
    realistic 'hedge at implied vol' benchmark.

    Returns
    -------
    dict[str, HedgeResult]
        Keys: 'gbm_vg', 'gbm_bsm', 'vg_vg', 'vg_bsm'.
    """
    p = dict(VG_PARAMS_DEFAULT if vg_params is None else vg_params)
    m = dict(MARKET if market is None else market)
    sigma, theta, nu = p["sigma"], p["theta"], p["nu"]
    r, q = m["r"], m["q"]

    T = horizon_days / HEDGE["trading_days"]
    K = S0
    rng = np.random.default_rng(seed)

    # premium & implied ATM vol from the VG model (the 'market' price)
    premium = float(vg_call_price(K, S0, T, r, q, sigma, theta, nu))
    atm_iv = implied_vol(premium, K, S0, T, r, q, "C")

    print(f"ATM call K={K:.0f}, T={T:.4f}y: VG premium={premium:.2f}, "
          f"implied vol={atm_iv:.4f}")

    # paths
    gbm = simulate_gbm_paths(S0, atm_iv, T, horizon_days, n_paths, r, q, rng)
    vgp = simulate_vg_paths(S0, sigma, theta, nu, T, horizon_days, n_paths,
                            r, q, rng)

    # delta tables
    print("Building delta lattices ...", flush=True)
    vg_table = build_vg_delta_table(K, S0, T, horizon_days, r, q,
                                    sigma, theta, nu)
    bsm_table = build_bsm_delta_table(K, S0, T, horizon_days, r, q, atm_iv)

    out = {
        "gbm_vg": run_hedge(gbm, vg_table, K, premium, T, r, "VG delta / GBM path"),
        "gbm_bsm": run_hedge(gbm, bsm_table, K, premium, T, r, "BSM delta / GBM path"),
        "vg_vg": run_hedge(vgp, vg_table, K, premium, T, r, "VG delta / VG path"),
        "vg_bsm": run_hedge(vgp, bsm_table, K, premium, T, r, "BSM delta / VG path"),
    }
    return out


def summarize(results: dict[str, HedgeResult]) -> str:
    """Format a results table as text."""
    lines = [f"{'scenario':<24}{'MAE':>10}{'RMSE':>10}{'mean':>10}"
             f"{'std':>10}{'Sharpe':>9}{'avg cost':>10}"]
    for key, res in results.items():
        he = res.hedge_error
        lines.append(f"{res.label:<24}{res.mae:>10.2f}{res.rmse:>10.2f}"
                     f"{he.mean():>10.2f}{he.std():>10.2f}{res.sharpe:>9.2f}"
                     f"{res.costs.mean():>10.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_experiment()
    print("\n" + summarize(results))
