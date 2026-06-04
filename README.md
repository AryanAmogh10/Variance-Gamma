# Variance-Gamma Options Pricing Engine

Pricing NIFTY 50 / BANK NIFTY index options with the **Variance-Gamma (VG)**
Lévy model — a fat-tailed, skew-aware alternative to Black-Scholes — calibrated
to **real NSE market data** and validated out-of-sample.

The VG model reproduces the volatility **skew** that a single Black-Scholes
volatility cannot, and does so with a consistent, arbitrage-free stochastic
process described by just three parameters.

---

## Highlights

- **Beats textbook Black-Scholes on real data.** On out-of-sample (held-out)
  strikes, calibrated VG reduces option-pricing RMSE by **~20%** versus a flat
  Black-Scholes model and wins on **~89% of independent trading days.**
- **Validated, not overfit.** Held-out-strike cross-validation shows an
  out-of-sample/in-sample error ratio of **≈1.01** — the model generalises to
  strikes it never saw.
- **Fast, stable pricing.** Whole option chains priced in one pass via the
  **Carr–Madan FFT** using the VG characteristic function (no fragile numerical
  integration bounds). Verified against the analytic Black-Scholes limit to
  ~0.001%.
- **Full historical dataset.** ~13M rows of daily NIFTY & BANK NIFTY option
  prices (every strike, every expiry) from options inception to present.

---

## The dataset

Daily OHLC / settle / open-interest for **every NIFTY and BANK NIFTY option**
(calls and puts, all strikes, all expiries):

| Index | Coverage |
|-------|----------|
| NIFTY 50  | 04-Jun-2001 → present |
| BANK NIFTY | 13-Jun-2005 → present |

📦 **Download on Kaggle:** _[add your Kaggle dataset link here]_

The data files are **not** stored in this repo (they're large) — grab them from
Kaggle and drop `NIFTY_options.csv` / `BANKNIFTY_options.csv` in the repo root.

Scrapers used to build the dataset are included:
`fetch_fo_history.py` (NSE historical API) and `fetch_bhavcopy.py` (NSE
bhavcopy archives for the earliest years).

---

## The engine — `vg_engine/`

| Module | Purpose |
|--------|---------|
| `vg_fft.py`     | VG characteristic function + Carr–Madan FFT option pricer |
| `bsm.py`        | Black-Scholes pricing + implied-volatility inversion |
| `data.py`       | Fast loaders (pandas + Parquet) for the option dataset |
| `calibrate.py`  | Calibrate VG `(σ, θ, ν)` to a live market option chain |
| `run_demo.py`   | End-to-end demo: calibrate to a real chain, compare to Black-Scholes |
| `overfit_check.py` | Out-of-sample / multi-day robustness validation |
| `backtest.py`   | Daily delta-hedging simulation across history |
| `_selftest.py`  | Correctness checks (martingale, put–call parity, VG→BSM limit) |

---

## Quick start

```bash
pip install -r requirements.txt

# 1) verify the pricer is correct
cd vg_engine && python _selftest.py

# 2) (one-time) convert the CSV to Parquet for fast loads
python run_demo.py NIFTY 2025-10-31 --build-parquet

# 3) calibrate VG to a real chain and compare to Black-Scholes
python run_demo.py NIFTY 2025-10-31

# 4) check it generalises out-of-sample
python overfit_check.py NIFTY 40
```

---

## Where it excels

- **Capturing the volatility skew / smile** — the systematic richness of OTM
  puts that flat Black-Scholes misses.
- **Tail-risk & return-distribution modelling** — fat tails and asymmetry make
  VG a strong fit for index log-returns, useful for VaR / risk work.
- **Parsimonious surface description** — one consistent three-parameter Lévy
  process across all strikes, rather than a separate fit per option.
- **Research & education** — a clean, tested reference implementation of VG
  pricing, calibration and FFT methods on a real, large-scale market dataset.

---

## Model

Risk-neutral Variance-Gamma (Madan–Carr–Chang, 1998):

```
S_T = S_0 · exp( (r − q + ω)·T + X_T ),    X_T ~ VG(σ, θ, ν)
ω   = (1/ν)·ln(1 − θν − ½σ²ν)              (martingale correction)
```

Options are priced via the Carr–Madan FFT of the VG characteristic function;
parameters are calibrated to market prices subject to the admissibility
constraint `1 − θν − ½σ²ν > 0`.

---

## Roadmap

- Walk-forward calibration producing a full historical parameter time series
- VG Greeks via FFT differentiation
- Broader Lévy family (NIG, CGMY) and a VG–GARCH hybrid
- VaR / Expected-Shortfall tail-risk backtesting
