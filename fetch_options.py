"""
fetch_options.py
----------------
Scrape the live NIFTY (or BANKNIFTY / FINNIFTY) option chain from NSE's public
REST endpoint and write it as a flat CSV the variance-gamma code can consume.

Reality check (important):
  * NSE has NO public websocket. The README's "web-socket" is loose wording.
    A true real-time tick/option feed needs a BROKER api (Zerodha Kite, Angel
    SmartAPI, Fyers, ...) with your own credentials.
  * What IS public is this REST endpoint, which returns a SNAPSHOT of the whole
    chain at the moment you call it:
        https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY
  * It is cookie/header gated and rate-limited. We open the homepage first to
    pick up cookies, send browser-like headers, and retry. If NSE blocks you,
    wait a bit and rerun, or run from an Indian IP.

To build a HISTORY of snapshots, run this on a schedule (e.g. every few minutes
during market hours). Each run appends a timestamped CSV.

Output columns:
    fetch_time, expiry, strike, type(CE/PE), spot, ltp, iv,
    oi, change_in_oi, volume, bid, ask

Usage:
    python fetch_options.py                         # NIFTY, nearest expiry
    python fetch_options.py --symbol BANKNIFTY
    python fetch_options.py --all-expiries
    python fetch_options.py --out nifty_chain.csv
"""

import argparse
import csv
import os
import time
import datetime
import requests

BASE = "https://www.nseindia.com"
API = BASE + "/api/option-chain-indices?symbol={symbol}"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE + "/option-chain",
    "Connection": "keep-alive",
}


def make_session() -> requests.Session:
    """Build a session primed with NSE cookies (needed or the API 401s/403s)."""
    s = requests.Session()
    s.headers.update(HEADERS)
    # Warm up: homepage then the option-chain page so we collect the cookies NSE
    # expects on the API call.
    s.get(BASE, timeout=10)
    s.get(BASE + "/option-chain", timeout=10)
    return s


def fetch_chain(symbol: str, retries: int = 4, pause: float = 2.0) -> dict:
    """Return the parsed JSON payload, retrying on transient blocks."""
    url = API.format(symbol=symbol.upper())
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            s = make_session()
            r = s.get(url, timeout=15)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                return r.json()
            last_err = f"HTTP {r.status_code}"
        except Exception as e:                       # noqa: BLE001
            last_err = str(e)
        print(f"  attempt {attempt}/{retries} failed ({last_err}); retrying ...",
              flush=True)
        time.sleep(pause * attempt)                  # back off
    raise RuntimeError(f"Could not fetch {symbol} option chain: {last_err}")


def flatten(payload: dict, only_nearest: bool) -> tuple[list[dict], float]:
    """Turn the nested NSE payload into flat per-strike-per-side rows."""
    records = payload["records"]
    spot = records.get("underlyingValue")
    all_expiries = records.get("expiryDates", [])
    keep_expiry = {all_expiries[0]} if (only_nearest and all_expiries) else None

    fetch_time = datetime.datetime.now().isoformat(timespec="seconds")
    rows: list[dict] = []

    for item in records["data"]:
        expiry = item.get("expiryDate")
        if keep_expiry is not None and expiry not in keep_expiry:
            continue
        strike = item.get("strikePrice")
        for side in ("CE", "PE"):
            leg = item.get(side)
            if not leg:
                continue
            rows.append({
                "fetch_time": fetch_time,
                "expiry": expiry,
                "strike": strike,
                "type": side,
                "spot": spot,
                "ltp": leg.get("lastPrice"),
                "iv": leg.get("impliedVolatility"),
                "oi": leg.get("openInterest"),
                "change_in_oi": leg.get("changeinOpenInterest"),
                "volume": leg.get("totalTradedVolume"),
                "bid": leg.get("bidprice"),
                "ask": leg.get("askPrice"),
            })
    return rows, spot


FIELDS = ["fetch_time", "expiry", "strike", "type", "spot", "ltp", "iv",
          "oi", "change_in_oi", "volume", "bid", "ask"]


def write_csv(rows: list[dict], out: str) -> None:
    """Append rows to CSV (write header only if the file is new/empty)."""
    new_file = not os.path.exists(out) or os.path.getsize(out) == 0
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch a live NSE index option chain.")
    p.add_argument("--symbol", default="NIFTY", help="NIFTY, BANKNIFTY, FINNIFTY")
    p.add_argument("--all-expiries", action="store_true",
                   help="Keep every expiry (default: nearest only)")
    p.add_argument("--out", default=None, help="Output CSV (default <symbol>_options.csv)")
    args = p.parse_args()

    out = args.out or f"{args.symbol.upper()}_options.csv"
    print(f"Fetching {args.symbol.upper()} option chain ...", flush=True)

    payload = fetch_chain(args.symbol)
    rows, spot = flatten(payload, only_nearest=not args.all_expiries)
    write_csv(rows, out)

    print(f"Spot {args.symbol.upper()} = {spot}")
    print(f"Appended {len(rows)} option rows to {out}")


if __name__ == "__main__":
    main()
