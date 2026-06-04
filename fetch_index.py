"""
fetch_index.py
--------------
Pull NIFTY 50 index OHLCV candles from Yahoo Finance and write them in the
CandleStick CSV format ("open,high,low,close,volume,timestamp") so they can be
read straight back by candlestick.readCandles().

This is the supported, reliable source for the *underlying* series that the
variance-gamma model fits its log-returns on (see vg.fit / vg.fit_moments).

NOTE on intraday history: Yahoo only serves ~60 days of <1d interval data, so
for "5m" the start date is clamped to today-59. For long history use "1d".

Usage:
    python fetch_index.py                      # default: ^NSEI, 5m, -> NIFTY__5m.txt
    python fetch_index.py --interval 1d --period 5y
    python fetch_index.py --symbol ^NSEBANK --interval 5m --out BANKNIFTY__5m.txt
"""

import argparse
import datetime
import yfinance as yf

from candlestick import CandleStick


def last_working_date() -> datetime.date:
    """Most recent weekday (skip Sat/Sun). Does not account for NSE holidays."""
    today = datetime.date.today()
    if today.weekday() == 5:        # Saturday
        return today - datetime.timedelta(days=1)
    if today.weekday() == 6:        # Sunday
        return today - datetime.timedelta(days=2)
    return today


def write_candles(file_name: str, data, mode: str = "w") -> int:
    """Write a yfinance OHLCV DataFrame as CandleStick lines. Returns row count."""
    written = 0
    with open(file_name, mode) as f:
        for i in range(len(data)):
            row = data.iloc[i]
            candle = CandleStick(
                round(float(row["Open"]), 6),
                round(float(row["High"]), 6),
                round(float(row["Low"]), 6),
                round(float(row["Close"]), 6),
                float(row["Volume"]),
                row.name,                      # the timestamp index
            )
            f.write(str(candle) + "\n")
            written += 1
    return written


def fetch_index(symbol: str, interval: str, out: str,
                period: str | None, days: int = 59) -> None:
    if interval.endswith("m") or interval.endswith("h"):
        # Intraday: Yahoo only keeps ~60 days, so use a start date window.
        start = (last_working_date() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        end = (last_working_date() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Fetching {symbol} {interval} from {start} to {end} ...", flush=True)
        data = yf.download(symbol, interval=interval, start=start, end=end,
                           progress=True, repair=True, auto_adjust=True)
    else:
        # Daily/weekly: use a period (e.g. 2y, 5y, max).
        period = period or "2y"
        print(f"Fetching {symbol} {interval} period={period} ...", flush=True)
        data = yf.download(symbol, interval=interval, period=period,
                           progress=True, repair=True, auto_adjust=True)

    if data is None or len(data) == 0:
        print("No data returned. Check the symbol / interval / your connection.")
        return

    # yfinance can return MultiIndex columns when a single symbol is requested;
    # flatten so row["Open"] etc. work regardless.
    if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
        data.columns = data.columns.get_level_values(0)

    n = write_candles(out, data, "w")
    print(f"Wrote {n} candles to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch NIFTY index OHLCV into CandleStick format.")
    p.add_argument("--symbol", default="^NSEI", help="Yahoo symbol (default ^NSEI = NIFTY 50)")
    p.add_argument("--interval", default="5m", help="5m, 15m, 1h, 1d, 1wk ...")
    p.add_argument("--period", default=None, help="For daily+ intervals: 1y, 2y, 5y, max")
    p.add_argument("--out", default=None, help="Output file (default NIFTY__<interval>.txt)")
    args = p.parse_args()

    out = args.out or f"NIFTY__{args.interval}.txt"
    fetch_index(args.symbol, args.interval, out, args.period)


if __name__ == "__main__":
    main()
