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

📦 **Download on Kaggle:**
https://www.kaggle.com/datasets/amogharyan10/nifty-daily-options-data-apr-2026
https://www.kaggle.com/datasets/amogharyan10/banknifty-daily-options-data-jun-2026

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
| `config.py`     | Central configuration (bump sizes, FFT, hedging, market params) |
| `greeks.py`     | Vectorised VG Greeks (delta, vega, volga, vanna, theta) + surface visualiser |
| `hedging.py`    | Monte-Carlo delta-hedging simulator (GBM & VG paths, VG vs BSM delta) |
| `run_analysis.py` | Generates all Greeks/hedging figures and summary stats |

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

## Greeks & Hedging

The engine includes a full **Greeks suite** and a **delta-hedging simulator**
(see `research_memo.md` for the complete study).

**Greeks** (`vg_engine/greeks.py`) — delta, vega, volga, vanna and theta via
central finite differences on the FFT pricer, with adaptive bump sizes from a
central config. The FFT prices a whole strike chain per pass, so a full
strike × tenor Greeks surface costs only ~11 FFT passes per tenor. Validated
by 17 unit tests: sign conventions, dividend-adjusted delta parity
(Δ_C − Δ_P = e^(−qT)), vega/volga/vanna call–put parity, bump-refinement
stability, and exact convergence to the closed-form Black-Scholes delta in the
Brownian limit.

**Hedging simulator** (`vg_engine/hedging.py`) — shorts an ATM call and
delta-hedges daily with 2 bps transaction costs along 1,000 GBM and 1,000 VG
Monte-Carlo paths, hedging with either the VG delta or the BSM delta at
implied vol. Deltas are precomputed on spot × time lattices and interpolated,
so the full 2×2 experiment runs in minutes. The P&L accounting (premium
accretion at r, financing, daily dividend crediting at q) is verified
bias-free: with costs off, the mean hedge error is statistically zero;
with costs on, the drag is ~13 points (≈2.6% of premium) per 30-day episode.

Headline finding: under fat-tailed VG dynamics, hedging-error dispersion is
~4–5× larger than under GBM *regardless of the hedge ratio* — jump risk is
structurally unhedgeable with the underlying alone, so the VG model's edge
lives in pricing and risk measurement rather than vanilla replication.

### Results

**Pricing: VG vs Black-Scholes on real NSE chains** (27 independent NIFTY
trading days 2021–2025, held-out strikes — the model is evaluated only on
strikes it never saw):

| Model | In-sample RMSE | Out-of-sample RMSE | OOS/IS ratio |
|---|---|---|---|
| **VG (3 params)** | 17.45 | **17.60** | 1.01 |
| BSM flat (1 param) | 24.40 | 25.37 | 1.04 |

VG beats flat BSM on **88.9%** of days with a median out-of-sample RMSE
reduction of **19.5%**, and the ≈1.01 OOS/IS ratio shows it is not overfitting.

**Hedging: terminal error of a daily-rebalanced short ATM call**
(K = 23700, 30 days, 1,000 Monte-Carlo paths per cell, 2 bps costs,
index points):

| Hedge / Paths | MAE | Mean | Std | Skew | 5th pct | 95th pct |
|---|---|---|---|---|---|---|
| VG delta / GBM paths | 73.9 | −12.1 | 99.2 | −1.19 | −190.4 | 119.0 |
| BSM delta / GBM paths | 49.7 | −13.3 | 65.5 | +0.01 | −119.3 | 97.1 |
| VG delta / VG paths | 267.9 | −12.2 | 396.7 | −2.80 | −769.7 | 307.1 |
| BSM delta / VG paths | 237.7 | −13.7 | 361.4 | −3.01 | −669.3 | 306.0 |

Moving from GBM to VG paths multiplies hedging-error dispersion ~4–5× for
*both* hedgers, and the VG-path error is heavily left-skewed — the loss tail
is where the jump risk lives. The ≈ −13 mean is pure transaction-cost drag
(with costs off the mean error is statistically zero).

**Calibrated VG parameters** (mean ± std across the 27 days): σ = 0.133 ± 0.029,
θ = −0.150 ± 0.092, ν = 0.114 ± 0.074 — persistently negative θ (skew) and
positive ν (fat tails), confirming both effects are structural in NIFTY options.

**Full write-ups:**
- [`research_memo.md`](research_memo.md) — the complete study (methodology,
  results, trading implications, limitations)
- [`validation_report.md`](validation_report.md) — every number explicit for
  external review (configs, error distributions, Greeks tables, parity
  checks to 1e-14, verbatim test output)
- [`figures/`](figures) — Greeks surfaces, hedging-error distributions,
  P&L confidence bands, delta-gap trajectories

```bash
# run the test suite (26 tests)
python -m pytest tests/ -v

# regenerate all figures + stats (figures/, seeded & reproducible)
cd vg_engine && python run_analysis.py

# regenerate the validation report
cd vg_engine && python make_validation_report.py
```

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
- Minimum-variance (variance-optimal) hedge ratios under VG
- Broader Lévy family (NIG, CGMY) and a VG–GARCH hybrid
- VaR / Expected-Shortfall tail-risk backtesting
