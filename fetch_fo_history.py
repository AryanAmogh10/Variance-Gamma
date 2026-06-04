"""
fetch_fo_history.py
-------------------
Bulk-download the FULL daily options history for NIFTY and BANKNIFTY (both CE and
PE) from NSE's official historical F&O endpoint:

    https://www.nseindia.com/api/historicalOR/foCPV
        ?from=DD-MM-YYYY&to=DD-MM-YYYY
        &instrumentType=OPTIDX&symbol=NIFTY&year=YYYY&optionType=CE

Each record is one contract on one trading day (daily OHLC / settle / OI / volume).

Design notes (why it is built this way):
  * 90-DAY CAP: the endpoint serves at most ~90 days per call, so we walk history
    in <=90-day windows.
  * `year` = the TRADE YEAR (the year of the from/to dates), NOT the expiry year.
    Confirmed empirically: a query for year=2026 returns contracts expiring as far
    out as 2029 -- i.e. a single window returns ALL expiries/strikes trading in that
    window. So we simply cover every trade date once, walking <=90-day windows
    *within each calendar year* (year param = that calendar year). Windows are
    non-overlapping and years are disjoint by trade date -> no duplicates, clean
    resume, and full coverage of every expiry.
  * RESUMABLE: every completed (symbol, optionType, from, to, year) is logged to a
    checkpoint file and skipped on re-run. Kill it any time; just run it again.
  * POLITE + ROBUST: primes NSE cookies, browser headers, throttles, retries with
    backoff, refreshes the session periodically.
  * SCHEMA-AGNOSTIC: columns are taken from the first non-empty response and locked
    in a sidecar .cols file, so we don't hardcode (and possibly mis-name) fields.

START DATES (NSE):  NIFTY options 2001-06-04 ; BANKNIFTY options 2005-06-13.
Older years may simply return empty from this API -> handled as "skip".

USAGE
    # 1) PROBE first -- pull one window, dump raw JSON so we can confirm fields:
    python fetch_fo_history.py --probe --symbol NIFTY --optionType CE \
        --from 28-05-2026 --to 04-06-2026 --year 2026

    # 2) FULL run (resumable -- safe to stop/restart):
    python fetch_fo_history.py

    # narrower / faster:
    python fetch_fo_history.py --symbols NIFTY --optionTypes CE PE \
        --start 01-01-2020 --year-lookahead 1 --delay 1.0
"""

import argparse
import csv
import json
import os
import time
import datetime as dt

import requests

BASE = "https://www.nseindia.com"
API = BASE + "/api/historicalOR/foCPV"

OUT_DIR = "fo_history"
RAW_DIR = os.path.join(OUT_DIR, "raw")
DONE_FILE = os.path.join(OUT_DIR, "_done.txt")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE + "/report-detail/fo_eq_security",
    "Connection": "keep-alive",
}

# Official NSE F&O start dates (DD-MM-YYYY).
DEFAULT_START = {
    "NIFTY": dt.date(2001, 6, 4),
    "BANKNIFTY": dt.date(2005, 6, 13),
}

DATEFMT = "%d-%m-%Y"


# --------------------------------------------------------------------------- #
# Session / fetching
# --------------------------------------------------------------------------- #
def make_session() -> requests.Session:
    """A session primed with NSE cookies (required or the API returns 401/403)."""
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get(BASE, timeout=10)
    s.get(BASE + "/report-detail/fo_eq_security", timeout=10)
    return s


def fetch_window(session: requests.Session, symbol: str, option_type: str,
                 d_from: dt.date, d_to: dt.date, year: int,
                 retries: int = 4, pause: float = 2.0):
    """Fetch one (symbol, optionType, window, year). Returns list[dict] (maybe empty).
    Returns the (possibly refreshed) session as well."""
    params = {
        "from": d_from.strftime(DATEFMT),
        "to": d_to.strftime(DATEFMT),
        "instrumentType": "OPTIDX",
        "symbol": symbol,
        "year": str(year),
        "optionType": option_type,
        # CRITICAL: the plain endpoint hard-caps at 70 rows. The CSV-export
        # variant returns the FULL result set (still JSON with our Accept header).
        "csv": "true",
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(API, params=params, timeout=20)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                return r.json().get("data", []), session
            last_err = f"HTTP {r.status_code}"
        except Exception as e:                       # noqa: BLE001
            last_err = str(e)
        # back off and rebuild the session (cookies may have expired / been blocked)
        time.sleep(pause * attempt)
        try:
            session = make_session()
        except Exception:                            # noqa: BLE001
            pass
    raise RuntimeError(f"Failed {symbol}/{option_type} {params['from']}->{params['to']} "
                       f"year={year}: {last_err}")


# --------------------------------------------------------------------------- #
# Output (schema-agnostic CSV with a locked column order per symbol+side)
# --------------------------------------------------------------------------- #
def cols_path(symbol: str, option_type: str) -> str:
    return os.path.join(OUT_DIR, f"{symbol}_{option_type}.cols")


def csv_path(symbol: str, option_type: str) -> str:
    return os.path.join(OUT_DIR, f"{symbol}_{option_type}.csv")


def load_cols(symbol: str, option_type: str):
    p = cols_path(symbol, option_type)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_cols(symbol: str, option_type: str, cols):
    with open(cols_path(symbol, option_type), "w") as f:
        json.dump(cols, f)


def append_rows(symbol: str, option_type: str, rows: list[dict]):
    if not rows:
        return
    cols = load_cols(symbol, option_type)
    path = csv_path(symbol, option_type)
    new_file = cols is None
    if new_file:
        cols = list(rows[0].keys())          # lock schema from first real response
        save_cols(symbol, option_type, cols)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Checkpoint / resume
# --------------------------------------------------------------------------- #
def load_done() -> set:
    if not os.path.exists(DONE_FILE):
        return set()
    with open(DONE_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def mark_done(key: str):
    with open(DONE_FILE, "a") as f:
        f.write(key + "\n")


# --------------------------------------------------------------------------- #
# Window iteration
# --------------------------------------------------------------------------- #
def windows(start: dt.date, end: dt.date, size_days: int = 90):
    cur = start
    step = dt.timedelta(days=size_days - 1)   # inclusive window of <=90 days
    one = dt.timedelta(days=1)
    while cur <= end:
        w_end = min(cur + step, end)
        yield cur, w_end
        cur = w_end + one


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_probe(args):
    os.makedirs(RAW_DIR, exist_ok=True)
    d_from = dt.datetime.strptime(args.from_, DATEFMT).date()
    d_to = dt.datetime.strptime(args.to, DATEFMT).date()
    s = make_session()
    rows, _ = fetch_window(s, args.symbol, args.optionType, d_from, d_to, args.year)
    raw = os.path.join(RAW_DIR, f"PROBE_{args.symbol}_{args.optionType}_{args.year}.json")
    with open(raw, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Got {len(rows)} rows. Raw JSON -> {raw}")
    if rows:
        print("Fields in each record:")
        for k in rows[0].keys():
            print(f"   {k}: {rows[0][k]!r}")
    else:
        print("Empty response for this window/year (try a different year or range).")


def run_full(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    today = dt.date(2026, 6, 4) if args.today is None \
        else dt.datetime.strptime(args.today, DATEFMT).date()

    done = load_done()
    session = make_session()
    requests_made = 0

    for symbol in args.symbols:
        start = DEFAULT_START.get(symbol, dt.date(2001, 1, 1))
        if args.start:
            start = dt.datetime.strptime(args.start, DATEFMT).date()

        # Walk <=90-day windows *within each calendar year* so the `year` param
        # always matches the trade dates. One query returns all expiries/strikes.
        for cal_year in range(start.year, today.year + 1):
            y_start = max(start, dt.date(cal_year, 1, 1))
            y_end = min(today, dt.date(cal_year, 12, 31))
            if y_start > y_end:
                continue

            for d_from, d_to in windows(y_start, y_end, 90):
                for option_type in args.optionTypes:
                    key = f"{symbol}|{option_type}|{d_from}|{d_to}|{cal_year}"
                    if key in done:
                        continue
                    try:
                        rows, session = fetch_window(
                            session, symbol, option_type, d_from, d_to, cal_year)
                    except RuntimeError as e:
                        print(f"  !! {e}")
                        continue

                    append_rows(symbol, option_type, rows)
                    mark_done(key)
                    requests_made += 1
                    print(f"[{requests_made}] {symbol} {option_type} "
                          f"{d_from}->{d_to} y{cal_year}: {len(rows)} rows")

                    time.sleep(args.delay)
                    # periodic session refresh to keep cookies fresh
                    if requests_made % args.refresh_every == 0:
                        session = make_session()

    print("\nDone. Output CSVs are in:", OUT_DIR)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Scrape full NSE F&O options history.")
    p.add_argument("--probe", action="store_true", help="Fetch one window and dump raw JSON")
    # probe args
    p.add_argument("--symbol", default="NIFTY")
    p.add_argument("--optionType", default="CE", choices=["CE", "PE"])
    p.add_argument("--from", dest="from_", default="28-05-2026", help="DD-MM-YYYY")
    p.add_argument("--to", default="04-06-2026", help="DD-MM-YYYY")
    p.add_argument("--year", type=int, default=2026)
    # full-run args
    p.add_argument("--symbols", nargs="+", default=["NIFTY", "BANKNIFTY"])
    p.add_argument("--optionTypes", nargs="+", default=["CE", "PE"])
    p.add_argument("--start", default=None, help="Override start date DD-MM-YYYY")
    p.add_argument("--today", default=None, help="Override end date DD-MM-YYYY")
    p.add_argument("--year-lookahead", type=int, default=3, dest="year_lookahead",
                   help="Extra expiry years to query per window (default 3)")
    p.add_argument("--delay", type=float, default=1.2, help="Seconds between requests")
    p.add_argument("--refresh-every", type=int, default=50, dest="refresh_every",
                   help="Rebuild session every N requests")
    args = p.parse_args()

    if args.probe:
        run_probe(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
