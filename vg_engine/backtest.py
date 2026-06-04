"""
backtest.py
-----------
VG vs Black-Scholes DELTA-HEDGING backtest -- the dynamic test of model quality.

Static price-fitting favours interpolators (a quadratic smile beats VG there).
But hedging requires a consistent *process*: you must produce a delta every day
and rebalance. This is where a real Levy model can beat BSM and where a smile
interpolator can't compete at all (it has no dynamics).

Experiment (per trade):
  * On an entry day, SELL an ATM option of the nearest monthly expiry at its
    market price, and delta-hedge it daily until expiry.
  * Compute deltas two ways:
       - VG    : params calibrated to that day's chain, delta by finite-diff repricing
       - BSM   : delta at the option's entry implied volatility
  * Track the self-financing hedge portfolio. At expiry the leftover cash is the
    REPLICATION ERROR (ideally 0). The better hedging model leaves a tighter,
    smaller error distribution.

Metric: mean |error|, std(error), RMSE -- in absolute points and as % of premium.

    python backtest.py [SYMBOL] [N_TRADES]
"""

import sys
import numpy as np
import pandas as pd

import data as D
from bsm import bsm_price, implied_vol
from vg_fft import vg_call_price, vg_put_price
from calibrate import calibrate_vg

R_FREE, DIV_Y = 0.066, 0.012


def vg_price(K, S, T, sigma, theta, nu, typ):
    f = vg_call_price if typ == "C" else vg_put_price
    return f(K, S, T, R_FREE, DIV_Y, sigma, theta, nu)


def vg_delta(K, S, T, sigma, theta, nu, typ, h=None):
    h = h or S * 0.005
    up = vg_price(K, S + h, T, sigma, theta, nu, typ)
    dn = vg_price(K, S - h, T, sigma, theta, nu, typ)
    return (up - dn) / (2 * h)


def bsm_delta(K, S, T, vol, typ):
    from scipy.stats import norm
    if T <= 0 or vol <= 0:
        if typ == "C":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (R_FREE - DIV_Y + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if typ == "C":
        return np.exp(-DIV_Y * T) * norm.cdf(d1)
    return -np.exp(-DIV_Y * T) * norm.cdf(-d1)


def clean_otm(sl, spot, T):
    sl = sl[((sl["type"] == "CE") & (sl["strike"] >= spot)) |
            ((sl["type"] == "PE") & (sl["strike"] <= spot))].copy()
    sl = sl[sl["close"] > 0]
    if len(sl) < 10:
        return None
    sl = sl[sl["volume"] >= max(sl["volume"].quantile(0.40), 1)]
    sl["typ"] = np.where(sl["type"].values == "CE", "C", "P")
    sl["iv"] = [implied_vol(p, K, spot, T, R_FREE, DIV_Y, t)
                for p, K, t in zip(sl["close"], sl["strike"], sl["typ"])]
    sl = sl[np.isfinite(sl["iv"])].sort_values("strike")
    return sl if len(sl) >= 12 else None


def spot_path(symbol, entry, expiry):
    """Daily underlying path between entry and expiry from Parquet."""
    df = pd.read_parquet(
        D.parquet_path(symbol),
        columns=["date", "spot"],
        filters=[("date", ">=", pd.Timestamp(entry)),
                 ("date", "<=", pd.Timestamp(expiry))],
    )
    df = df[df["spot"] > 0].dropna()
    s = df.groupby("date")["spot"].first().sort_index()
    return s


def hedge_one(path_dates, path_spots, K, typ, premium, delta_fn):
    """Simulate daily delta-hedging of a SHORT option. Returns replication error."""
    S = path_spots
    dates = path_dates
    # t0
    d0 = delta_fn(S[0], _years(dates[0], dates[-1]))
    cash = premium - d0 * S[0]        # sold option, bought d0 shares
    prev_delta = d0
    for i in range(1, len(S)):
        dt = _years(dates[i - 1], dates[i])
        cash *= np.exp(R_FREE * dt)                      # finance cash
        cash += prev_delta * S[i - 1] * (np.exp(DIV_Y * dt) - 1)  # dividends on shares
        Trem = _years(dates[i], dates[-1])
        d = delta_fn(S[i], Trem) if Trem > 1 / 365 else prev_delta
        cash -= (d - prev_delta) * S[i]                  # rebalance
        prev_delta = d
    # expiry: liquidate shares, pay payoff
    ST = S[-1]
    payoff = max(ST - K, 0.0) if typ == "C" else max(K - ST, 0.0)
    cash += prev_delta * ST - payoff
    return cash


def _years(d0, d1):
    return (pd.Timestamp(d1) - pd.Timestamp(d0)).days / 365.0


def run_trade(symbol, date):
    chain = D.load_chain(symbol, date, liquid_only=True)
    if len(chain) == 0:
        return None
    spot = D.get_spot(chain)
    if not np.isfinite(spot):
        return None

    exps = (chain.groupby("expiry")
            .agg(n=("strike", "size"), vol=("volume", "sum"), T=("T", "first"))
            .reset_index())
    exps = exps[(exps["T"] >= 0.05) & (exps["T"] <= 0.15) & (exps["n"] >= 20)]
    if exps.empty:
        return None
    expiry = exps.sort_values("vol", ascending=False).iloc[0]["expiry"]
    sl = D.expiry_slice(chain, expiry)
    T = float(sl["T"].iloc[0])

    cl = clean_otm(sl, spot, T)
    if cl is None:
        return None

    # calibrate VG on this day's chain
    cal = calibrate_vg(spot, T, R_FREE, DIV_Y,
                       cl["strike"].values, cl["close"].values, cl["typ"].values)
    if not cal["success"]:
        return None

    # pick the ATM call to hedge
    sl_c = sl[(sl["type"] == "CE") & (sl["close"] > 0) & (sl["volume"] > 0)].copy()
    if sl_c.empty:
        return None
    sl_c["dist"] = (sl_c["strike"] - spot).abs()
    opt = sl_c.sort_values("dist").iloc[0]
    K, premium, typ = float(opt["strike"]), float(opt["close"]), "C"
    iv0 = implied_vol(premium, K, spot, T, R_FREE, DIV_Y, typ)
    if not np.isfinite(iv0):
        return None

    # build the spot path to expiry
    s = spot_path(symbol, date, str(pd.Timestamp(expiry).date()))
    if len(s) < 5 or s.index[-1] < pd.Timestamp(expiry) - pd.Timedelta(days=5):
        return None
    dates = list(s.index)
    spots = s.values.astype(float)

    sig, th, nu = cal["sigma"], cal["theta"], cal["nu"]
    err_vg = hedge_one(dates, spots, K, typ, premium,
                       lambda S, Tr: vg_delta(K, S, max(Tr, 1e-4), sig, th, nu, typ))
    err_bsm = hedge_one(dates, spots, K, typ, premium,
                        lambda S, Tr: bsm_delta(K, S, max(Tr, 1e-4), iv0, typ))

    return {"date": date, "K": K, "premium": premium, "T": T,
            "err_vg": err_vg, "err_bsm": err_bsm}


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NIFTY"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    dates_all = pd.read_parquet(D.parquet_path(symbol), columns=["date"])["date"].dropna().unique()
    dates_all = np.sort(pd.to_datetime(dates_all))
    mask = (dates_all >= pd.Timestamp("2021-01-01")) & (dates_all <= pd.Timestamp("2025-08-31"))
    pool = dates_all[mask]
    pick = pool[np.linspace(0, len(pool) - 1, n).astype(int)]

    rows = []
    for d in pick:
        try:
            r = run_trade(symbol, str(pd.Timestamp(d).date()))
        except Exception:
            r = None
        if r:
            rows.append(r)
            print(f"  {r['date']}  K={r['K']:.0f} prem={r['premium']:7.2f}  "
                  f"|err| VG={abs(r['err_vg']):7.2f}  BSM={abs(r['err_bsm']):7.2f}",
                  flush=True)

    if not rows:
        print("No trades."); return
    df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print(f"DELTA-HEDGING BACKTEST  ({len(df)} trades, {symbol})")
    print("=" * 60)

    def stats(col):
        e = df[col].values
        return (np.mean(np.abs(e)), np.std(e), np.sqrt(np.mean(e ** 2)),
                np.mean(np.abs(e) / df["premium"].values) * 100)

    print(f"\n  {'Model':<8}{'mean|err|':>11}{'std(err)':>11}{'RMSE':>10}{'|err|/prem':>12}")
    for name, col in [("VG", "err_vg"), ("BSM", "err_bsm")]:
        m, s, r, p = stats(col)
        print(f"  {name:<8}{m:>11.2f}{s:>11.2f}{r:>10.2f}{p:>11.1f}%")

    vg_win = (df["err_vg"].abs() < df["err_bsm"].abs()).mean() * 100
    rmse_red = (1 - np.sqrt(np.mean(df["err_vg"]**2)) / np.sqrt(np.mean(df["err_bsm"]**2))) * 100
    print(f"\n  VG hedges tighter on {vg_win:.1f}% of trades")
    print(f"  VG reduces hedging RMSE by {rmse_red:.1f}% vs BSM")


if __name__ == "__main__":
    main()
