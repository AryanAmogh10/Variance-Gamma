"""
fetch_bhavcopy.py
-----------------
Downloads NSE F&O bhavcopy zip files for a date range, extracts daily OHLC data
for NIFTY and BANKNIFTY options (OPTIDX), and appends rows in the same column
format as fetch_fo_history.py so the two datasets can be concatenated cleanly.

Bhavcopy URL pattern:
  https://www.nseindia.com/content/historical/DERIVATIVES/YYYY/MMM/fo{DD}{MMM}{YYYY}bhav.csv.zip

Bhavcopy CSV columns (vary slightly by era, handled dynamically):
  INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP,
  OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH,
  OPEN_INT, CHG_IN_OI, TIMESTAMP

Output columns match existing CSVs:
  FH_INSTRUMENT, FH_SYMBOL, FH_EXPIRY_DT, FH_STRIKE_PRICE, FH_OPTION_TYPE,
  FH_MARKET_TYPE, FH_OPENING_PRICE, FH_TRADE_HIGH_PRICE, FH_TRADE_LOW_PRICE,
  FH_CLOSING_PRICE, FH_LAST_TRADED_PRICE, FH_PREV_CLS, FH_SETTLE_PRICE,
  FH_TOT_TRADED_QTY, FH_TOT_TRADED_VAL, FH_OPEN_INT, FH_CHANGE_IN_OI,
  FH_MARKET_LOT, FH_TIMESTAMP, FH_TIMESTAMP_ORDER, FH_UNDERLYING_VALUE,
  CALCULATED_PREMIUM_VAL
"""

import csv
import datetime
import io
import os
import time
import zipfile
import requests

OUT_DIR = "fo_history"
DONE_FILE = os.path.join(OUT_DIR, "_bhav_done.txt")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}

MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN",
          "JUL","AUG","SEP","OCT","NOV","DEC"]

OUTPUT_FIELDS = [
    "FH_INSTRUMENT","FH_SYMBOL","FH_EXPIRY_DT","FH_STRIKE_PRICE","FH_OPTION_TYPE",
    "FH_MARKET_TYPE","FH_OPENING_PRICE","FH_TRADE_HIGH_PRICE","FH_TRADE_LOW_PRICE",
    "FH_CLOSING_PRICE","FH_LAST_TRADED_PRICE","FH_PREV_CLS","FH_SETTLE_PRICE",
    "FH_TOT_TRADED_QTY","FH_TOT_TRADED_VAL","FH_OPEN_INT","FH_CHANGE_IN_OI",
    "FH_MARKET_LOT","FH_TIMESTAMP","FH_TIMESTAMP_ORDER","FH_UNDERLYING_VALUE",
    "CALCULATED_PREMIUM_VAL"
]

SYMBOLS = {"NIFTY", "BANKNIFTY"}


def bhav_url(date: datetime.date) -> str:
    dd  = date.strftime("%d")
    mmm = MONTHS[date.month - 1]
    yyyy = date.strftime("%Y")
    return (f"https://archives.nseindia.com/content/historical/DERIVATIVES/"
            f"{yyyy}/{mmm}/fo{dd}{mmm}{yyyy}bhav.csv.zip")


def load_done() -> set:
    if not os.path.exists(DONE_FILE):
        return set()
    with open(DONE_FILE) as f:
        return set(l.strip() for l in f if l.strip())


def mark_done(key: str):
    with open(DONE_FILE, "a") as f:
        f.write(key + "\n")


def normalize_row(row: dict, date: datetime.date) -> dict | None:
    """Map a bhavcopy row to output schema. Returns None if not an options row we want."""
    instrument = row.get("INSTRUMENT","").strip()
    symbol     = row.get("SYMBOL","").strip()
    opt_type   = row.get("OPTION_TYP","").strip()

    if instrument != "OPTIDX":
        return None
    if symbol not in SYMBOLS:
        return None
    if opt_type not in ("CE", "PE"):
        return None

    def f(k, default="0"):
        return row.get(k, default).strip() or default

    expiry_raw = f("EXPIRY_DT")   # e.g. "29-JUN-2001" or "29-Jun-2001"
    timestamp  = date.strftime("%d-%b-%Y")   # e.g. "04-Jun-2001"

    return {
        "FH_INSTRUMENT":       "OPTIDX",
        "FH_SYMBOL":           symbol,
        "FH_EXPIRY_DT":        expiry_raw,
        "FH_STRIKE_PRICE":     f("STRIKE_PR"),
        "FH_OPTION_TYPE":      opt_type,
        "FH_MARKET_TYPE":      "N",
        "FH_OPENING_PRICE":    f("OPEN"),
        "FH_TRADE_HIGH_PRICE": f("HIGH"),
        "FH_TRADE_LOW_PRICE":  f("LOW"),
        "FH_CLOSING_PRICE":    f("CLOSE"),
        "FH_LAST_TRADED_PRICE":"0",
        "FH_PREV_CLS":         "0",
        "FH_SETTLE_PRICE":     f("SETTLE_PR"),
        "FH_TOT_TRADED_QTY":   f("CONTRACTS"),
        "FH_TOT_TRADED_VAL":   f("VAL_INLAKH"),
        "FH_OPEN_INT":         f("OPEN_INT"),
        "FH_CHANGE_IN_OI":     f("CHG_IN_OI"),
        "FH_MARKET_LOT":       "0",
        "FH_TIMESTAMP":        timestamp,
        "FH_TIMESTAMP_ORDER":  "",
        "FH_UNDERLYING_VALUE": "0",
        "CALCULATED_PREMIUM_VAL": "0",
    }


def fetch_bhav(date: datetime.date, session: requests.Session):
    """Download and parse one day's bhavcopy. Returns list of matching rows."""
    url = bhav_url(date)
    for attempt in range(1, 4):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 404:
                return None   # holiday / non-trading day
            if r.status_code == 200:
                break
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * attempt)
    else:
        return None

    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            text = f.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    rows = []
    reader = csv.DictReader(io.StringIO(text))
    # strip whitespace from headers
    reader.fieldnames = [h.strip() if h else "__blank__"
                         for h in (reader.fieldnames or [])]
    for row in reader:
        row = {k.strip(): v for k, v in row.items() if k and k.strip()}
        norm = normalize_row(row, date)
        if norm:
            rows.append(norm)
    return rows


def append_to_csv(symbol: str, rows: list[dict]):
    path = os.path.join("fo_history", f"{symbol}_bhav.csv")
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerows(rows)


def trading_days(start: datetime.date, end: datetime.date):
    cur = start
    while cur <= end:
        if cur.weekday() < 5:   # Mon-Fri only (NSE holidays handled by 404)
            yield cur
        cur += datetime.timedelta(days=1)


def run(ranges: list[tuple]):
    os.makedirs(OUT_DIR, exist_ok=True)
    done = load_done()
    s = requests.Session()
    s.headers.update(HEADERS)

    total = 0
    for start, end in ranges:
        for date in trading_days(start, end):
            key = date.isoformat()
            if key in done:
                continue

            rows = fetch_bhav(date, s)
            if rows is None:
                # non-trading day or 404 — mark done to skip next time
                mark_done(key)
                continue

            # split by symbol
            by_sym = {}
            for r in rows:
                by_sym.setdefault(r["FH_SYMBOL"], []).append(r)

            for sym, sym_rows in by_sym.items():
                append_to_csv(sym, sym_rows)

            mark_done(key)
            total += len(rows)
            print(f"{date}  {len(rows):4d} rows  (total so far: {total:,})")
            time.sleep(0.4)

    print(f"\nDone. {total:,} rows written.")


if __name__ == "__main__":
    # Gap 1: NIFTY options started Jun 2001, API only goes back to Apr 2002
    # Gap 2: BANKNIFTY options started Jun 2005, API only goes back to Apr 2008
    # Gap 3: Both missing Nov 2025 onwards
    GAPS = [
        (datetime.date(2001, 6,  4), datetime.date(2002, 3, 31)),   # NIFTY early history
        (datetime.date(2005, 6, 13), datetime.date(2008, 3, 31)),   # BANKNIFTY early history
        # Nov 2025+ returns 404 from archives -- genuinely not published yet
    ]
    run(GAPS)
