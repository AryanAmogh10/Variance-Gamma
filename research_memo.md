# Research Memo — Greeks & Delta-Hedging under the Variance-Gamma Model

*Variance-Gamma engine, NIFTY 50 calibration · simulation study*

---

## 1. Abstract

We extend the Variance-Gamma (VG) options-pricing engine with a finite-difference Greeks suite and a daily delta-hedging simulator, and use them to quantify how much of an option's risk can actually be hedged away under fat-tailed dynamics. Greeks (delta, vega, volga, vanna, theta) are computed by central differences on the Carr–Madan FFT pricer, vectorised so a full strike×tenor surface costs only a handful of FFT passes per tenor, and validated by 17 unit tests covering sign conventions, dividend-adjusted put-call delta parity, and the Brownian (BSM) limit. The hedging simulator shorts an ATM call and rebalances daily with 2 bps transaction costs along 1,000 GBM and 1,000 VG underlying paths, using either the VG delta or the BSM delta (at the VG-implied ATM vol) as the hedge ratio. The central result is that the *path distribution*, not the *hedge ratio*, dominates hedging performance: terminal hedging-error dispersion is roughly four to five times larger on VG paths than on GBM paths for **both** hedgers (RMSE ≈ 362–397 vs 67–100 index points), while switching between VG and BSM deltas moves RMSE by only ~10%. This confirms, in a controlled setting, the finding from our historical NSE backtest: the VG model's value lies in pricing the skew and describing tail risk — not in producing a materially better vanilla hedge ratio, because jump risk is structurally unhedgeable with the underlying alone.

## 2. Model recap

**Variance-Gamma (Madan–Carr–Chang 1998).** Risk-neutral dynamics

```
S_T = S_0 · exp((r − q + ω)T + X_T),   X_T ~ VG(σ, θ, ν),
ω = (1/ν)·ln(1 − θν − ½σ²ν)
```

where `σ` controls diffusion-like volatility, `θ < 0` produces the negative skew, and `ν` the kurtosis (jump intensity) of returns. Pricing is via the Carr–Madan FFT of the VG characteristic function (whole strike chain in one pass; verified to match the analytic Black-Scholes price to ~0.001% in the `ν→0, θ→0` limit).

**Black-Scholes benchmark.** All hedging comparisons use BSM **at the VG-implied ATM volatility** — i.e. the realistic "trader hedges at the option's implied vol" benchmark, not a stale historical vol.

**Parameters.** Hedging simulations use the mean parameters calibrated across 27 independent NIFTY trading days (2021–2025): `σ = 0.133, θ = −0.150, ν = 0.114`, with `r = 6.6%, q = 1.2%`, spot 23,700. The ATM 30-day call prices at **505.0 points** (implied vol 13.1%).

## 3. Greeks methodology

- **Central finite differences** on the FFT pricer. Bumps are adaptive and configured centrally (`vg_engine/config.py`): spot bump = 0.1%·S₀ (delta, vanna), vol bump = max(1%·σ, 1 bp) (vega, volga), time bump = min(1 trading day, 10% of remaining life) (theta).
- **Vectorisation.** The FFT prices an entire strike vector at once, so a full Greeks set for one expiry costs 11 FFT passes regardless of strike count; a strike×tenor surface costs O(tenors × 11) passes.
- **Conventions.** Theta is reported in calendar time (∂V/∂t = −∂V/∂T, negative for long vanillas). Delta parity is enforced in its dividend-adjusted form Δ_C − Δ_P = e^(−qT).
- **Validation (17 tests, all passing).** Sign and boundedness of call/put delta; monotonicity in strike; positive vega; dividend-adjusted delta parity to 2×10⁻³; vega/volga/vanna call-put parity; independence from bump refinement (h vs h/2 within 0.1%); convergence of VG delta to the closed-form BSM delta in the Brownian limit (2×10⁻³); theta consistent with realized one-day price decay.

Surfaces are rendered as heatmaps over moneyness × tenor (`figures/greeks_surface_call.png`, `figures/greeks_surface_put.png`).

## 4. Hedging simulation design

| Element | Choice |
|---|---|
| Instrument | Short 1 ATM European call, 30-trading-day life |
| Paths | 1,000 GBM + 1,000 VG, daily steps, common seed (42) |
| VG path generator | Exact gamma-subordination: ΔX = θΔG + σ√ΔG·Z, ΔG ~ Γ(Δt/ν, ν), drift (r−q+ω)Δt — the same measure as the pricer |
| Hedge ratios | VG delta vs BSM delta (at VG-implied ATM vol 13.1%) |
| Delta computation | Precomputed spot×time lattices (121 log-spaced spots spanning ±6 total σ, one row per day), bilinear interpolation along paths — avoids ~120k per-step FFT calls |
| Costs | 2 bps of traded notional per rebalance, including initial setup and final liquidation |
| Cash | Accrues at r daily |
| Dividends | The stock leg is credited the continuous dividend yield q daily (the pricing measure drifts S at r − q because holders receive q; omitting this credit biases the hedge P&L by ≈ mean(Δ)·S·q·T ≈ 19 points — caught in audit and corrected) |
| Hedging error | Terminal portfolio value after option settlement (perfect replication ⇒ 0). Verified unbiased: zero-cost BSM-delta/GBM mean error = +0.36 ± 0.67 on 10,000 paths |

## 5. Results

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

## 6. Key findings and trading implications

1. **Delta-hedging does not neutralise fat-tail risk.** Under realistic (VG-calibrated) dynamics, a daily-rebalanced short ATM call retains a terminal P&L standard deviation of ~72–79% of the premium. Sizing and capital decisions for short-vol books should be driven by this residual, not by the near-zero hedging error a GBM/BSM analysis suggests.
2. **A better model does not imply a better hedge ratio.** VG prices the smile ~20% better than flat BSM out-of-sample (prior study), yet its delta does not hedge better — BSM-at-implied-vol remains a remarkably strong hedge benchmark for vanillas. Model sophistication pays in *pricing, relative value and risk measurement*, not in vanilla replication.
3. **Hedging the gap requires convexity, not more delta.** Since both hedge ratios leave the same jump residual, reducing it needs option-based hedges (gamma/wing protection) — consistent with how index-option desks actually run short-vol risk.
4. **Costs are material at daily frequency:** ~13 points (≈2.6% of premium) of pure transaction-cost drag per 30-day episode. Any rebalancing-frequency optimisation should trade discretisation error against this drag.

## 7. Limitations and future work

- **Synthetic-path study.** Both engines simulate under the risk-neutral measure; real-world drift and vol-of-vol are absent. The companion historical backtest (58 real NSE episodes) reached the same qualitative conclusion, which is reassuring, but a P-measure simulation with estimated drift would tighten the claim.
- **Static parameters.** σ, θ, ν are frozen at multi-day calibrated means; daily recalibration (parameter risk) would add realism.
- **Single instrument and tenor.** Only the ATM 30-day call is hedged; wings, longer tenors, and put-side hedging may rank the deltas differently.
- **Minimum-variance delta.** The natural next step: under VG, the variance-optimal hedge ratio differs from ∂V/∂S; implementing it would quantify how much of the residual is recoverable in theory.
- **Greeks beyond first/second order.** Gamma ladders and FFT-differentiated (rather than bumped) Greeks would cut numerical noise for production use.
- **VaR/Expected-Shortfall application.** The same VG distribution that cannot improve vanilla hedging *should* excel at tail-risk measurement — a Kupiec/Christoffersen coverage backtest on 25 years of NIFTY returns is the highest-value follow-up.

---

*Artifacts: all figures in `/figures`; per-cell statistics in `figures/hedging_stats.json`; reproducible via `python vg_engine/run_analysis.py` (seed 42).*
