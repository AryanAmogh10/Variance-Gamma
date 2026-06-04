# vg_engine

A Variance-Gamma options-pricing & calibration engine built on the scraped
NIFTY/BANKNIFTY option history. This upgrades the original project from
*fitting a distribution to historical index returns* to *calibrating a
risk-neutral Lévy model to real market option prices and benchmarking it
against Black-Scholes*.

## Modules

| File | Purpose |
|------|---------|
| `vg_fft.py` | VG characteristic function + **Carr-Madan FFT** pricer (prices the whole strike chain at once; replaces the old `scipy.quad` pricer with hardcoded integration bounds) |
| `bsm.py` | Black-Scholes pricing + implied-vol inversion (the benchmark) |
| `data.py` | Fast loaders for the 1 GB option CSVs (pandas + optional Parquet) |
| `calibrate.py` | Calibrate `(sigma, theta, nu)` to a market option chain; benchmark VG vs flat BSM |
| `run_demo.py` | End-to-end demo on a real NIFTY chain |
| `_selftest.py` | Correctness tests (martingale, put-call parity, **VG→BSM in the nu→0 limit**, IV round-trip) |

## Quick start

```bash
pip install -r ../requirements.txt

# verify the pricer is correct
python _selftest.py

# (optional, one-time) convert CSV -> Parquet for fast loads
python run_demo.py NIFTY 2025-10-31 --build-parquet

# calibrate VG to a real chain and compare against Black-Scholes
python run_demo.py NIFTY 2025-10-31
```

## What the demo shows

On a liquid monthly NIFTY expiry, after cleaning stale/illiquid prints
(volume floor + put-call-parity/IV outlier removal), **calibrated VG reduces
option-pricing RMSE by ~23% and MAPE from ~52% to ~17% versus a flat
Black-Scholes model** — because VG captures the volatility skew that a single
BSM vol cannot.

## Model

Risk-neutral VG (Madan-Carr-Chang 1998):

    S_T = S_0 * exp((r - q + omega) T + X_T),   X_T ~ VG(sigma, theta, nu)
    omega = (1/nu) ln(1 - theta*nu - 0.5 sigma^2 nu)   (martingale correction)

Pricing uses the Carr-Madan FFT; calibration minimises relative pricing error
on the OTM wings subject to the admissibility constraint
`1 - theta*nu - 0.5 sigma^2 nu > 0`.

## Next steps (not yet built)

- Walk-forward calibration across the full history (per-day surface)
- Greeks under VG (FFT differentiation) + a VG-vs-BSM delta-hedging backtest
- Broader Lévy family (NIG, CGMY) and a VG-GARCH hybrid
- Risk-free term structure (currently a flat 6.6%) and per-date dividend yield
