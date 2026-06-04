"""
overfit_check.py
----------------
Rigorously test whether the VG-beats-BSM result is real or just overfitting.

Three tests:
  1. HELD-OUT-STRIKE CV: calibrate VG on a train subset of strikes, evaluate
     pricing error on the held-out strikes it never saw. Small in-sample vs
     out-of-sample gap => not overfitting.
  2. FAIR PARAM COUNT: compare VG (3 params) to a quadratic IV smile (3 params),
     not only to flat BSM (1 param). If VG wins OOS against an equal-parameter
     curve, the edge is not just "more knobs".
  3. MULTI-DAY ROBUSTNESS: repeat across many independent trading days; report
     win-rate and parameter stability.

    python overfit_check.py [SYMBOL] [N_DAYS]
"""

import sys
import numpy as np
import pandas as pd

import data as D
from bsm import bsm_price, implied_vol
from calibrate import calibrate_vg

R_FREE, DIV_Y = 0.066, 0.012


# --------------------------------------------------------------------------- #
def clean_otm(chain, spot, T):
    """Return cleaned OTM (strike, price, type, iv) for one expiry."""
    sl = chain[((chain["type"] == "CE") & (chain["strike"] >= spot)) |
               ((chain["type"] == "PE") & (chain["strike"] <= spot))].copy()
    sl = sl[sl["close"] > 0]
    if len(sl) < 10:
        return None
    vol_floor = max(sl["volume"].quantile(0.40), 1)
    sl = sl[sl["volume"] >= vol_floor]
    sl["typ"] = np.where(sl["type"].values == "CE", "C", "P")
    sl["iv"] = [implied_vol(p, K, spot, T, R_FREE, DIV_Y, t)
                for p, K, t in zip(sl["close"], sl["strike"], sl["typ"])]
    sl = sl[np.isfinite(sl["iv"])].sort_values("strike")
    if len(sl) < 12:
        return None
    med = sl["iv"].rolling(7, center=True, min_periods=3).median()
    mad = (sl["iv"] - med).abs().rolling(7, center=True, min_periods=3).median()
    sl = sl[((sl["iv"] - med).abs() <= (5 * mad + 0.02)).fillna(True)]
    return sl if len(sl) >= 12 else None


def quad_smile_fit(logm, iv):
    """Fit IV = a + b*logm + c*logm^2 (3 params). Returns coeffs."""
    A = np.vstack([np.ones_like(logm), logm, logm ** 2]).T
    coef, *_ = np.linalg.lstsq(A, iv, rcond=None)
    return coef


def quad_smile_iv(coef, logm):
    return coef[0] + coef[1] * logm + coef[2] * logm ** 2


def rmse(model, mkt):
    return float(np.sqrt(np.mean((np.asarray(model) - np.asarray(mkt)) ** 2)))


# --------------------------------------------------------------------------- #
def analyze_day(symbol, date):
    """Held-out-strike CV for the most liquid monthly-ish expiry on `date`."""
    chain = D.load_chain(symbol, date, liquid_only=True)
    if len(chain) == 0:
        return None
    spot = D.get_spot(chain)
    if not np.isfinite(spot):
        return None

    exps = (chain.groupby("expiry")
            .agg(n=("strike", "size"), vol=("volume", "sum"), T=("T", "first"))
            .reset_index())
    exps = exps[(exps["T"] >= 0.04) & (exps["T"] <= 0.25) & (exps["n"] >= 20)]
    if exps.empty:
        return None
    expiry = exps.sort_values("vol", ascending=False).iloc[0]["expiry"]
    sl = D.expiry_slice(chain, expiry)
    T = float(sl["T"].iloc[0])

    cl = clean_otm(sl, spot, T)
    if cl is None:
        return None

    cl = cl.reset_index(drop=True)
    strikes = cl["strike"].values.astype(float)
    prices = cl["close"].values.astype(float)
    types = cl["typ"].values
    ivs = cl["iv"].values
    logm = np.log(strikes / spot)

    # TRAIN/TEST split: alternate strikes (test strikes lie BETWEEN train strikes
    # so models must interpolate the smile, not extrapolate)
    idx = np.arange(len(strikes))
    train, test = idx[::2], idx[1::2]
    if len(test) < 4 or len(train) < 6:
        return None

    # ---- VG: calibrate on train, evaluate on test ----
    cal = calibrate_vg(spot, T, R_FREE, DIV_Y,
                       strikes[train], prices[train], types[train])
    from vg_fft import vg_call_price, vg_put_price

    def vg_px(K, t):
        f = vg_call_price if t == "C" else vg_put_price
        return f(K, spot, T, R_FREE, DIV_Y, cal["sigma"], cal["theta"], cal["nu"])

    vg_is = rmse([vg_px(K, t) for K, t in zip(strikes[train], types[train])], prices[train])
    vg_oos = rmse([vg_px(K, t) for K, t in zip(strikes[test], types[test])], prices[test])

    # ---- flat BSM (1 param): ATM vol from train ----
    atm_vol = ivs[train][np.argmin(np.abs(strikes[train] - spot))]
    bsm_is = rmse([bsm_price(K, spot, T, R_FREE, DIV_Y, atm_vol, t)
                   for K, t in zip(strikes[train], types[train])], prices[train])
    bsm_oos = rmse([bsm_price(K, spot, T, R_FREE, DIV_Y, atm_vol, t)
                    for K, t in zip(strikes[test], types[test])], prices[test])

    # ---- quadratic IV smile (3 params): fit on train, evaluate on test ----
    coef = quad_smile_fit(logm[train], ivs[train])
    q_is = rmse([bsm_price(K, spot, T, R_FREE, DIV_Y,
                           max(quad_smile_iv(coef, np.log(K/spot)), 1e-3), t)
                 for K, t in zip(strikes[train], types[train])], prices[train])
    q_oos = rmse([bsm_price(K, spot, T, R_FREE, DIV_Y,
                            max(quad_smile_iv(coef, np.log(K/spot)), 1e-3), t)
                  for K, t in zip(strikes[test], types[test])], prices[test])

    return {
        "date": date, "T": T, "n": len(strikes),
        "sigma": cal["sigma"], "theta": cal["theta"], "nu": cal["nu"],
        "vg_is": vg_is, "vg_oos": vg_oos,
        "bsm_is": bsm_is, "bsm_oos": bsm_oos,
        "quad_is": q_is, "quad_oos": q_oos,
    }


# --------------------------------------------------------------------------- #
def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NIFTY"
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    # sample trading days spread across recent history (last_dates of months etc.)
    pq = D.parquet_path(symbol)
    dates_all = pd.read_parquet(pq, columns=["date"])["date"].dropna().unique()
    dates_all = np.sort(pd.to_datetime(dates_all))
    # restrict to 2021-2025 (good liquidity) and sample evenly
    mask = (dates_all >= pd.Timestamp("2021-01-01")) & (dates_all <= pd.Timestamp("2025-10-31"))
    pool = dates_all[mask]
    pick = pool[np.linspace(0, len(pool) - 1, n_days).astype(int)]

    rows = []
    for d in pick:
        try:
            r = analyze_day(symbol, str(pd.Timestamp(d).date()))
        except Exception as e:
            r = None
        if r:
            rows.append(r)
            print(f"  {r['date']}  VG_oos={r['vg_oos']:7.2f}  "
                  f"BSM_oos={r['bsm_oos']:7.2f}  quad_oos={r['quad_oos']:7.2f}  "
                  f"(s={r['sigma']:.3f} th={r['theta']:+.3f} nu={r['nu']:.3f})",
                  flush=True)

    if not rows:
        print("No valid days."); return
    df = pd.DataFrame(rows)

    print("\n" + "=" * 64)
    print(f"OVERFIT ANALYSIS  ({len(df)} independent trading days, {symbol})")
    print("=" * 64)

    print("\n[1] In-sample vs Out-of-sample RMSE (held-out strikes)")
    print(f"  {'Model':<16}{'IS RMSE':>10}{'OOS RMSE':>10}{'OOS/IS':>9}")
    for name, isc, oosc in [("VG (3p)", "vg_is", "vg_oos"),
                            ("Quad smile (3p)", "quad_is", "quad_oos"),
                            ("BSM flat (1p)", "bsm_is", "bsm_oos")]:
        i, o = df[isc].mean(), df[oosc].mean()
        print(f"  {name:<16}{i:>10.2f}{o:>10.2f}{o/i:>9.2f}")

    print("\n[2] Out-of-sample win-rate (lower RMSE on UNSEEN strikes)")
    vg_beats_bsm = (df["vg_oos"] < df["bsm_oos"]).mean() * 100
    vg_beats_quad = (df["vg_oos"] < df["quad_oos"]).mean() * 100
    print(f"  VG beats flat BSM : {vg_beats_bsm:5.1f}% of days")
    print(f"  VG beats quad smile: {vg_beats_quad:5.1f}% of days")
    print(f"  Median OOS RMSE reduction VG vs BSM : "
          f"{(1 - (df['vg_oos']/df['bsm_oos'])).median()*100:5.1f}%")

    print("\n[3] Parameter stability across days (stable => structural, not noise)")
    for p in ["sigma", "theta", "nu"]:
        print(f"  {p:<6} mean={df[p].mean():+.4f}  std={df[p].std():.4f}  "
              f"CV={df[p].std()/abs(df[p].mean()):.2f}")

    print("\nInterpretation:")
    gap = df['vg_oos'].mean() / df['vg_is'].mean()
    if gap < 1.5 and vg_beats_bsm > 60:
        print("  VG generalises to unseen strikes and wins out-of-sample"
              " => NOT overfitting.")
    else:
        print("  Large IS/OOS gap or weak win-rate => caution, possible overfit.")


if __name__ == "__main__":
    main()
