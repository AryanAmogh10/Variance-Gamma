"""
make_validation_report.py
-------------------------
Generate validation_report.md for external review: all numbers explicit,
no chart references. Reruns the seeded hedging experiment (with and without
transaction costs), computes Greeks sample tables, runs pytest, and excerpts
the research memo.

    python make_validation_report.py
"""

from __future__ import annotations

import os
import subprocess
import sys

# Windows cp1252 consoles cannot print Δ etc. -- force UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy.stats import skew

from vg_fft import vg_call_price
from bsm import implied_vol
from greeks import greeks_chain
from hedging import (simulate_gbm_paths, simulate_vg_paths,
                     build_vg_delta_table, build_bsm_delta_table, run_hedge)
from config import HEDGE, MARKET, VG_PARAMS_DEFAULT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "validation_report.md")

S0 = 23700.0
K_ATM = S0
GREEK_STRIKES = np.array([20000.0, 22500.0, 23700.0, 25000.0, 27500.0])
GREEK_LABELS = ["deep ITM", "ITM", "ATM", "OTM", "deep OTM"]   # call-side labels


def run_experiment_full() -> dict:
    """Rerun the seeded 2x2 experiment, with tc=2bps and tc=0 on the SAME paths."""
    p, m = VG_PARAMS_DEFAULT, MARKET
    sigma, theta, nu = p["sigma"], p["theta"], p["nu"]
    r, q = m["r"], m["q"]
    n_paths, n_days = HEDGE["n_paths"], HEDGE["horizon_days"]
    T = n_days / HEDGE["trading_days"]
    rng = np.random.default_rng(HEDGE["seed"])

    premium = float(vg_call_price(K_ATM, S0, T, r, q, sigma, theta, nu))
    atm_iv = implied_vol(premium, K_ATM, S0, T, r, q, "C")

    gbm = simulate_gbm_paths(S0, atm_iv, T, n_days, n_paths, r, q, rng)
    vgp = simulate_vg_paths(S0, sigma, theta, nu, T, n_days, n_paths, r, q, rng)

    print("Building delta lattices ...", flush=True)
    vg_tab = build_vg_delta_table(K_ATM, S0, T, n_days, r, q, sigma, theta, nu)
    bs_tab = build_bsm_delta_table(K_ATM, S0, T, n_days, r, q, atm_iv)

    cells = {}
    for tc, tag in [(HEDGE["tc_bps"], "tc"), (0.0, "free")]:
        cells[("gbm_vg", tag)] = run_hedge(gbm, vg_tab, K_ATM, premium, T, r,
                                           "VG delta / GBM paths", tc_bps=tc)
        cells[("gbm_bsm", tag)] = run_hedge(gbm, bs_tab, K_ATM, premium, T, r,
                                            "BSM delta / GBM paths", tc_bps=tc)
        cells[("vg_vg", tag)] = run_hedge(vgp, vg_tab, K_ATM, premium, T, r,
                                          "VG delta / VG paths", tc_bps=tc)
        cells[("vg_bsm", tag)] = run_hedge(vgp, bs_tab, K_ATM, premium, T, r,
                                           "BSM delta / VG paths", tc_bps=tc)
    return {"cells": cells, "premium": premium, "atm_iv": atm_iv, "T": T}


def greeks_tables() -> tuple[str, str, str]:
    """Sample Greeks tables for calls and puts + delta-parity check."""
    p, m = VG_PARAMS_DEFAULT, MARKET
    T = HEDGE["horizon_days"] / HEDGE["trading_days"]
    c = greeks_chain(GREEK_STRIKES, "C", S0, T, m["r"], m["q"],
                     p["sigma"], p["theta"], p["nu"])
    pu = greeks_chain(GREEK_STRIKES, "P", S0, T, m["r"], m["q"],
                      p["sigma"], p["theta"], p["nu"])

    def table(g, typ):
        hdr = (f"| Strike | Class | Price | Delta | Vega | Vanna | Volga | Theta |\n"
               f"|---|---|---|---|---|---|---|---|\n")
        rows = []
        for i, (K, lab) in enumerate(zip(GREEK_STRIKES, GREEK_LABELS)):
            lab_t = lab if typ == "C" else GREEK_LABELS[len(GREEK_LABELS)-1-i]
            rows.append(f"| {K:.0f} | {lab_t} | {g['price'][i]:.2f} | "
                        f"{g['delta'][i]:+.4f} | {g['vega'][i]:.2f} | "
                        f"{g['vanna'][i]:+.4f} | {g['volga'][i]:.1f} | "
                        f"{g['theta'][i]:+.1f} |")
        return hdr + "\n".join(rows)

    parity_hdr = ("| Strike | Δ_call | Δ_put | Δ_call − Δ_put | exp(−qT) | abs diff |\n"
                  "|---|---|---|---|---|---|\n")
    eqt = np.exp(-m["q"] * T)
    prow = []
    for i, K in enumerate(GREEK_STRIKES):
        gap = c["delta"][i] - pu["delta"][i]
        prow.append(f"| {K:.0f} | {c['delta'][i]:+.4f} | {pu['delta'][i]:+.4f} | "
                    f"{gap:.4f} | {eqt:.4f} | {abs(gap-eqt):.2e} |")
    return table(c, "C"), table(pu, "P"), parity_hdr + "\n".join(prow)


def run_pytest() -> str:
    """Run the full test suite and capture verbatim output."""
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"],
                       capture_output=True, text=True, cwd=ROOT)
    return r.stdout + (r.stderr or "")


def memo_excerpts() -> tuple[str, str]:
    """Extract Results and Key Findings sections from research_memo.md."""
    with open(os.path.join(ROOT, "research_memo.md"), encoding="utf-8") as f:
        memo = f.read()
    res = memo.split("## 5. Results")[1].split("## 6.")[0].strip()
    kf = memo.split("## 6. Key findings and trading implications")[1] \
             .split("## 7.")[0].strip()
    return res, kf


def main() -> None:
    exp = run_experiment_full()
    cells, premium, atm_iv, T = (exp["cells"], exp["premium"],
                                 exp["atm_iv"], exp["T"])
    p, m, h = VG_PARAMS_DEFAULT, MARKET, HEDGE

    # --- section 2 table ---
    order = ["gbm_vg", "gbm_bsm", "vg_vg", "vg_bsm"]
    s2 = ("| Hedge / Paths | MAE | Mean | Std | Skew | 5th pct | 95th pct |\n"
          "|---|---|---|---|---|---|---|\n")
    for key in order:
        res = cells[(key, "tc")]
        he = res.hedge_error
        s2 += (f"| {res.label} | {res.mae:.2f} | {he.mean():.2f} | "
               f"{he.std():.2f} | {skew(he):+.3f} | "
               f"{np.percentile(he, 5):.2f} | {np.percentile(he, 95):.2f} |\n")

    # --- section 5: cost impact + rebalance count ---
    s5 = ("| Hedge / Paths | Mean P&L (2 bps) | Mean P&L (0 bps) | Cost drag |\n"
          "|---|---|---|---|\n")
    for key in order:
        a, b = cells[(key, "tc")], cells[(key, "free")]
        s5 += (f"| {a.label} | {a.hedge_error.mean():.2f} | "
               f"{b.hedge_error.mean():.2f} | "
               f"{a.hedge_error.mean() - b.hedge_error.mean():.2f} |\n")
    # rebalances: initial + (n_days-1) daily + final liquidation
    n_reb = 1 + (h["horizon_days"] - 1) + 1

    call_tab, put_tab, parity_tab = greeks_tables()
    pytest_out = run_pytest()
    res_x, kf_x = memo_excerpts()

    report = f"""# Validation Report — VG Greeks & Delta-Hedging Engine

Generated for external review. All numbers explicit; fully reproducible with
seed {h['seed']} via `python vg_engine/make_validation_report.py`.

---

## 1. Run Configuration

| Parameter | Value |
|---|---|
| Spot (S0) | {S0:.1f} |
| Hedged instrument | Short 1 ATM European call, strike K = {K_ATM:.0f} |
| Option life | {h['horizon_days']} trading days (T = {T:.4f} years, {h['trading_days']} days/yr) |
| Greeks strike range | {GREEK_STRIKES.min():.0f} – {GREEK_STRIKES.max():.0f} |
| VG sigma | {p['sigma']} |
| VG theta | {p['theta']} |
| VG nu | {p['nu']} |
| Risk-free rate r | {m['r']} |
| Dividend yield q | {m['q']} |
| ATM VG premium | {premium:.2f} index points |
| ATM implied vol (BSM-inverted) | {atm_iv:.4f} |
| Paths per engine | {h['n_paths']} GBM + {h['n_paths']} VG |
| Rebalance frequency | Daily ({h['horizon_days']} steps) |
| Transaction cost | {h['tc_bps']} bps of traded notional per rebalance |
| RNG seed | {h['seed']} |

VG parameters are the mean calibrated values across 27 independent NIFTY
trading days (2021–2025). The BSM hedger uses the VG-implied ATM vol.

---

## 2. Hedging Error Comparison

Terminal hedging error per path (index points), 1,000 paths per cell,
2 bps costs included. Perfect replication = 0.

{s2}

---

## 3. Greeks Sample Table

One expiry: T = {T:.4f} years ({h['horizon_days']} trading days). Spot = {S0:.0f}.
Theta is calendar-time dV/dt (negative = decay). Vega per unit of sigma.

### Calls

{call_tab}

### Puts

{put_tab}

### Put-call delta parity check

Expected: Δ_call − Δ_put = exp(−qT) = {np.exp(-m['q']*T):.4f}

{parity_tab}

---

## 4. Test Suite Results

```
{pytest_out.strip()}
```

---

## 5. Transaction Cost Impact

{s5}
Average number of rebalances per path: {n_reb} (1 initial hedge + {h['horizon_days']-1} daily rebalances + 1 final liquidation). The hedge trades every day by construction (delta always moves); the 2 bps cost applies to traded notional |Δshares| × S each day.

---

## 6. Memo Excerpts (verbatim)

### Results (from research_memo.md §5)

{res_x}

### Key findings and trading implications (from research_memo.md §6)

{kf_x}
"""

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {OUT}\n")
    print("=" * 78)
    print(report)


if __name__ == "__main__":
    main()
