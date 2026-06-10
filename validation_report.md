# Validation Report — VG Greeks & Delta-Hedging Engine

Generated for external review. All numbers explicit; fully reproducible with
seed 42 via `python vg_engine/make_validation_report.py`.

---

## 1. Run Configuration

| Parameter | Value |
|---|---|
| Spot (S0) | 23700.0 |
| Hedged instrument | Short 1 ATM European call, strike K = 23700 |
| Option life | 30 trading days (T = 0.1190 years, 252 days/yr) |
| Greeks strike range | 20000 – 27500 |
| VG sigma | 0.133 |
| VG theta | -0.15 |
| VG nu | 0.114 |
| Risk-free rate r | 0.066 |
| Dividend yield q | 0.012 |
| ATM VG premium | 505.02 index points |
| ATM implied vol (BSM-inverted) | 0.1309 |
| Paths per engine | 1000 GBM + 1000 VG |
| Rebalance frequency | Daily (30 steps) |
| Transaction cost | 2.0 bps of traded notional per rebalance |
| RNG seed | 42 |

VG parameters are the mean calibrated values across 27 independent NIFTY
trading days (2021–2025). The BSM hedger uses the VG-implied ATM vol.

---

## 2. Hedging Error Comparison

Terminal hedging error per path (index points), 1,000 paths per cell,
2 bps costs included. Perfect replication = 0.

| Hedge / Paths | MAE | Mean | Std | Skew | 5th pct | 95th pct |
|---|---|---|---|---|---|---|
| VG delta / GBM paths | 73.92 | -12.11 | 99.22 | -1.189 | -190.35 | 119.03 |
| BSM delta / GBM paths | 49.74 | -13.30 | 65.47 | +0.009 | -119.29 | 97.08 |
| VG delta / VG paths | 267.85 | -12.15 | 396.65 | -2.795 | -769.66 | 307.14 |
| BSM delta / VG paths | 237.66 | -13.68 | 361.40 | -3.007 | -669.32 | 305.98 |


---

## 3. Greeks Sample Table

One expiry: T = 0.1190 years (30 trading days). Spot = 23700.
Theta is calendar-time dV/dt (negative = decay). Vega per unit of sigma.

### Calls

| Strike | Class | Price | Delta | Vega | Vanna | Volga | Theta |
|---|---|---|---|---|---|---|---|
| 20000 | deep ITM | 3827.87 | +0.9933 | 154.96 | -0.1275 | 3162.2 | -1091.0 |
| 22500 | ITM | 1439.77 | +0.9004 | 1381.97 | -0.8414 | 8123.9 | -1950.7 |
| 23700 | ATM | 505.02 | +0.6455 | 2564.17 | -0.5866 | 3091.9 | -2763.3 |
| 25000 | OTM | 68.41 | +0.1171 | 1592.91 | +1.6110 | 14260.5 | -1026.5 |
| 27500 | deep OTM | 1.55 | +0.0027 | 93.32 | +0.1355 | 4047.1 | -33.4 |

### Puts

| Strike | Class | Price | Delta | Vega | Vanna | Volga | Theta |
|---|---|---|---|---|---|---|---|
| 20000 | deep OTM | 5.18 | -0.0052 | 154.96 | -0.1275 | 3162.2 | -65.3 |
| 22500 | OTM | 97.51 | -0.0981 | 1381.97 | -0.8414 | 8123.9 | -761.3 |
| 23700 | ATM | 353.37 | -0.3531 | 2564.17 | -0.5866 | 3091.9 | -1495.3 |
| 25000 | ITM | 1206.58 | -0.8815 | 1592.91 | +1.6110 | 14260.5 | +326.6 |
| 27500 | deep ITM | 3620.16 | -0.9959 | 93.32 | +0.1355 | 4047.1 | +1483.4 |

### Put-call delta parity check

Expected: Δ_call − Δ_put = exp(−qT) = 0.9986

| Strike | Δ_call | Δ_put | Δ_call − Δ_put | exp(−qT) | abs diff |
|---|---|---|---|---|---|
| 20000 | +0.9933 | -0.0052 | 0.9986 | 0.9986 | 5.05e-14 |
| 22500 | +0.9004 | -0.0981 | 0.9986 | 0.9986 | 1.68e-14 |
| 23700 | +0.6455 | -0.3531 | 0.9986 | 0.9986 | 3.83e-14 |
| 25000 | +0.1171 | -0.8815 | 0.9986 | 0.9986 | 4.36e-14 |
| 27500 | +0.0027 | -0.9959 | 0.9986 | 0.9986 | 6.13e-14 |

---

## 4. Test Suite Results

```
============================= test session starts =============================
platform win32 -- Python 3.13.13, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\amogh\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\amogh\OneDrive\Desktop\VG
plugins: anyio-4.12.0
collecting ... collected 26 items

tests/test_greeks.py::TestSigns::test_call_delta_positive_and_bounded PASSED [  3%]
tests/test_greeks.py::TestSigns::test_put_delta_negative_and_bounded PASSED [  7%]
tests/test_greeks.py::TestSigns::test_delta_monotone_decreasing_in_strike PASSED [ 11%]
tests/test_greeks.py::TestSigns::test_vega_positive PASSED               [ 15%]
tests/test_greeks.py::TestSigns::test_theta_negative_for_long_vanillas PASSED [ 19%]
tests/test_greeks.py::TestSigns::test_itm_call_delta_above_otm PASSED    [ 23%]
tests/test_greeks.py::TestParity::test_delta_parity_dividend_adjusted PASSED [ 26%]
tests/test_greeks.py::TestParity::test_delta_parity_no_dividend PASSED   [ 30%]
tests/test_greeks.py::TestParity::test_vega_parity PASSED                [ 34%]
tests/test_greeks.py::TestParity::test_volga_parity PASSED               [ 38%]
tests/test_greeks.py::TestParity::test_vanna_parity PASSED               [ 42%]
tests/test_greeks.py::TestFDConsistency::test_delta_matches_manual_fd PASSED [ 46%]
tests/test_greeks.py::TestFDConsistency::test_bump_refinement_stability PASSED [ 50%]
tests/test_greeks.py::TestFDConsistency::test_vg_delta_near_bsm_in_brownian_limit PASSED [ 53%]
tests/test_greeks.py::TestFDConsistency::test_theta_matches_price_decay PASSED [ 57%]
tests/test_greeks.py::TestSurface::test_surface_shapes PASSED            [ 61%]
tests/test_greeks.py::TestSurface::test_surface_rows_match_chain PASSED  [ 65%]
tests/test_hedging.py::TestPathSimulation::test_gbm_martingale PASSED    [ 69%]
tests/test_hedging.py::TestPathSimulation::test_vg_martingale PASSED     [ 73%]
tests/test_hedging.py::TestPathSimulation::test_vg_has_fatter_tails_than_gbm PASSED [ 76%]
tests/test_hedging.py::TestPathSimulation::test_paths_start_at_s0 PASSED [ 80%]
tests/test_hedging.py::TestPathSimulation::test_omega_admissible PASSED  [ 84%]
tests/test_hedging.py::TestHedging::test_bsm_hedge_on_gbm_replicates PASSED [ 88%]
tests/test_hedging.py::TestHedging::test_costs_reduce_pnl PASSED         [ 92%]
tests/test_hedging.py::TestHedging::test_delta_paths_bounded PASSED      [ 96%]
tests/test_hedging.py::TestHedging::test_result_shapes PASSED            [100%]

============================= 26 passed in 4.40s ==============================
```

---

## 5. Transaction Cost Impact

| Hedge / Paths | Mean P&L (2 bps) | Mean P&L (0 bps) | Cost drag |
|---|---|---|---|
| VG delta / GBM paths | -12.11 | 2.11 | -14.21 |
| BSM delta / GBM paths | -13.30 | 0.06 | -13.36 |
| VG delta / VG paths | -12.15 | -2.16 | -9.99 |
| BSM delta / VG paths | -13.68 | -4.32 | -9.36 |

Average number of rebalances per path: 31 (1 initial hedge + 29 daily rebalances + 1 final liquidation). The hedge trades every day by construction (delta always moves); the 2 bps cost applies to traded notional |Δshares| × S each day.

---

## 6. Memo Excerpts (verbatim)

### Results (from research_memo.md §5)

**Summary statistics (1,000 paths per cell, index points):**

| Scenario | MAE | RMSE | mean | std | Sharpe | avg cost |
|---|---|---|---|---|---|---|
| VG delta / GBM path | 73.9 | 100.0 | −12.1 | 99.2 | −0.35 | 14.2 |
| BSM delta / GBM path | 49.7 | 66.8 | −13.3 | 65.5 | −0.59 | 13.3 |
| VG delta / VG path | 267.9 | 396.8 | −12.2 | 396.7 | −0.09 | 10.0 |
| BSM delta / VG path | 237.7 | 361.7 | −13.7 | 361.4 | −0.11 | 9.3 |

**Reading the table:**

- **Path dynamics dominate.** Moving from GBM to VG paths multiplies hedging-error RMSE by ~4–5× for *both* hedge ratios. Daily delta-hedging nearly replicates under diffusion; under VG dynamics the gamma-time jumps create gap risk that no spot-only hedge can remove (`figures/hedge_error_dist_gbm_paths.png` vs `figures/hedge_error_dist_vg_paths.png`).
- **The hedge-ratio choice is second-order.** On VG paths, BSM delta modestly outperforms VG delta (RMSE 361 vs 395, ~9%). The two deltas differ most early in the option's life and converge as expiry approaches (`figures/delta_gap_over_time.png`, `figures/delta_convergence.png`).
- **The mean error ≈ −12 to −14 points on all four cells** is pure transaction-cost drag (avg cost 9–14 points): with costs switched off the mean hedge error is statistically zero (+0.36 ± 0.67 on 10,000 paths), confirming the P&L accounting — premium accretion, financing at r and dividend crediting at q — is unbiased.
- **P&L bands** (`figures/pnl_paths_confidence_bands.png`) show the VG-path 5–95% band widening steadily through the life of the trade — the signature of unhedgeable jump risk accumulating, rather than terminal-date noise.

### Key findings and trading implications (from research_memo.md §6)

1. **Delta-hedging does not neutralise fat-tail risk.** Under realistic (VG-calibrated) dynamics, a daily-rebalanced short ATM call retains a terminal P&L standard deviation of ~72–79% of the premium. Sizing and capital decisions for short-vol books should be driven by this residual, not by the near-zero hedging error a GBM/BSM analysis suggests.
2. **A better model does not imply a better hedge ratio.** VG prices the smile ~20% better than flat BSM out-of-sample (prior study), yet its delta does not hedge better — BSM-at-implied-vol remains a remarkably strong hedge benchmark for vanillas. Model sophistication pays in *pricing, relative value and risk measurement*, not in vanilla replication.
3. **Hedging the gap requires convexity, not more delta.** Since both hedge ratios leave the same jump residual, reducing it needs option-based hedges (gamma/wing protection) — consistent with how index-option desks actually run short-vol risk.
4. **Costs are material at daily frequency:** ~13 points (≈2.6% of premium) of pure transaction-cost drag per 30-day episode. Any rebalancing-frequency optimisation should trade discretisation error against this drag.
