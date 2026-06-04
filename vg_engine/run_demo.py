"""
run_demo.py
-----------
End-to-end demo: load a real NIFTY option chain, calibrate Variance-Gamma to it,
and compare VG vs flat Black-Scholes pricing error across the smile.

    python run_demo.py [SYMBOL] [DATE] [--build-parquet]
    e.g. python run_demo.py NIFTY 2025-10-31
"""

import sys
import numpy as np

import data as D
from calibrate import calibrate_vg, benchmark_vs_bsm
from bsm import implied_vol

# India risk-free ~ 91-day T-bill; NIFTY dividend yield ~ 1.2%
R_FREE = 0.066
DIV_Y = 0.012


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "NIFTY"
    date = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else "2025-10-31"

    if "--build-parquet" in sys.argv:
        print(f"Building Parquet for {symbol} ...")
        D.build_parquet(symbol)

    print(f"\nLoading {symbol} chain for {date} ...")
    chain = D.load_chain(symbol, date, liquid_only=True)
    spot = D.get_spot(chain)
    print(f"Spot = {spot:.2f},  {len(chain)} liquid contracts, "
          f"{chain['expiry'].nunique()} expiries")

    # pick the most LIQUID monthly-ish expiry (T in ~3 weeks to 3 months)
    exps = (chain.groupby("expiry")
            .agg(n=("strike", "size"), vol=("volume", "sum"), T=("T", "first"))
            .reset_index())
    exps = exps[(exps["T"] >= 0.04) & (exps["T"] <= 0.25) & (exps["n"] >= 15)]
    if exps.empty:
        print("No suitable expiry found."); return
    expiry = exps.sort_values("vol", ascending=False).iloc[0]["expiry"]
    sl = D.expiry_slice(chain, expiry)
    T = float(sl["T"].iloc[0])
    print(f"\nCalibrating expiry {expiry.date()}  (T={T:.4f}y, {len(sl)} contracts)")

    # OTM wings only: calls above spot, puts below spot (liquid & informative)
    sl = sl[((sl["type"] == "CE") & (sl["strike"] >= spot)) |
            ((sl["type"] == "PE") & (sl["strike"] <= spot))].copy()
    sl = sl[sl["close"] > 0]

    # ---- DATA CLEANING (stale/illiquid prints corrupt calibration) ----
    # 1) require real liquidity
    vol_floor = max(sl["volume"].quantile(0.40), 1)
    sl = sl[sl["volume"] >= vol_floor]
    # 2) compute market IV and drop non-invertible / arbitrage-violating prints
    sl["typ"] = np.where(sl["type"].values == "CE", "C", "P")
    sl["iv"] = [implied_vol(p, K, spot, T, R_FREE, DIV_Y, t)
                for p, K, t in zip(sl["close"], sl["strike"], sl["typ"])]
    sl = sl[np.isfinite(sl["iv"])]
    # 3) drop IV outliers vs local median (kills the parity-breaking stale ticks)
    sl = sl.sort_values("strike")
    med = sl["iv"].rolling(7, center=True, min_periods=3).median()
    mad = (sl["iv"] - med).abs().rolling(7, center=True, min_periods=3).median()
    keep = (sl["iv"] - med).abs() <= (5 * mad + 0.02)
    sl = sl[keep.fillna(True)]

    strikes = sl["strike"].values
    prices = sl["close"].values
    types = sl["typ"].values
    print(f"Using {len(strikes)} cleaned OTM contracts "
          f"(vol>={vol_floor:.0f}, strikes {strikes.min():.0f}-{strikes.max():.0f})")

    # ---- calibrate ----
    cal = calibrate_vg(spot, T, R_FREE, DIV_Y, strikes, prices, types)
    print("\n--- Calibrated VG parameters ---")
    print(f"  sigma = {cal['sigma']:.4f}   theta = {cal['theta']:.4f}   nu = {cal['nu']:.4f}")
    print(f"  fit RMSE = {cal['rmse']:.3f}   MAPE = {cal['mape']:.2f}%   (success={cal['success']})")

    # ---- benchmark vs flat BSM ----
    bm = benchmark_vs_bsm(spot, T, R_FREE, DIV_Y, strikes, prices, types, cal)
    print("\n--- VG vs flat Black-Scholes (ATM vol = {:.3f}) ---".format(bm["atm_vol"]))
    print(f"  {'Model':<12}{'RMSE (pts)':>14}{'MAPE (%)':>12}")
    print(f"  {'VG (smile)':<12}{bm['vg_rmse']:>14.3f}{bm['vg_mape']:>12.2f}")
    print(f"  {'BSM (flat)':<12}{bm['bsm_rmse']:>14.3f}{bm['bsm_mape']:>12.2f}")
    improve = (1 - bm["vg_rmse"] / bm["bsm_rmse"]) * 100
    print(f"\n  => VG reduces pricing RMSE by {improve:.1f}% vs flat BSM")

    # ---- show the smile fit ----
    print("\n--- Per-strike detail (market vs models) ---")
    print(f"  {'Strike':>8}{'Type':>5}{'Mkt':>10}{'VG':>10}{'BSM':>10}{'MktIV':>8}")
    order = np.argsort(strikes)
    for i in order:
        print(f"  {strikes[i]:>8.0f}{types[i]:>5}{prices[i]:>10.2f}"
              f"{bm['vg_prices'][i]:>10.2f}{bm['bsm_prices'][i]:>10.2f}"
              f"{bm['mkt_iv'][i]:>8.3f}")


if __name__ == "__main__":
    main()
