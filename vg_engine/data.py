"""
data.py
-------
Fast loaders for the scraped NIFTY/BANKNIFTY option history CSVs.

The raw CSVs are ~1 GB; pure-python csv loops choke on them. This module uses
pandas (+ optional Parquet) and returns tidy, typed option chains ready for
pricing/calibration.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

# default location of the consolidated CSVs (repo root, one level up)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_COLS = {
    "FH_SYMBOL": "symbol",
    "FH_EXPIRY_DT": "expiry",
    "FH_STRIKE_PRICE": "strike",
    "FH_OPTION_TYPE": "type",
    "FH_OPENING_PRICE": "open",
    "FH_TRADE_HIGH_PRICE": "high",
    "FH_TRADE_LOW_PRICE": "low",
    "FH_CLOSING_PRICE": "close",
    "FH_SETTLE_PRICE": "settle",
    "FH_TOT_TRADED_QTY": "volume",
    "FH_OPEN_INT": "oi",
    "FH_TIMESTAMP": "date",
    "FH_UNDERLYING_VALUE": "spot",
}


def csv_path(symbol: str) -> str:
    return os.path.join(ROOT, f"{symbol.upper()}_options.csv")


def parquet_path(symbol: str) -> str:
    return os.path.join(ROOT, f"{symbol.upper()}_options.parquet")


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RAW_COLS)[list(RAW_COLS.values())].copy()
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    df["expiry"] = pd.to_datetime(df["expiry"], format="%d-%b-%Y", errors="coerce")
    for c in ["strike", "open", "high", "low", "close", "settle",
              "volume", "oi", "spot"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["T"] = (df["expiry"] - df["date"]).dt.days / 365.0
    return df


def build_parquet(symbol: str, chunksize: int = 1_000_000) -> str:
    """One-time conversion CSV -> Parquet for fast repeated querying."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = parquet_path(symbol)
    writer = None
    rows = 0
    for chunk in pd.read_csv(csv_path(symbol), chunksize=chunksize,
                             low_memory=False):
        tidy = _tidy(chunk)
        table = pa.Table.from_pandas(tidy, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out, table.schema)
        writer.write_table(table)
        rows += len(tidy)
        print(f"  ...{rows:,} rows", flush=True)
    if writer:
        writer.close()
    print(f"Wrote {out} ({rows:,} rows)")
    return out


def load_chain(symbol: str, date: str, liquid_only: bool = True) -> pd.DataFrame:
    """
    Load one trade date's full option chain as a tidy DataFrame.

    Uses Parquet if available (fast), else chunk-scans the CSV.
    `date` accepts 'YYYY-MM-DD' or '31-Oct-2025'.
    `liquid_only` keeps only contracts that actually traded that day.
    """
    ts = pd.to_datetime(date)

    pq = parquet_path(symbol)
    if os.path.exists(pq):
        df = pd.read_parquet(pq, filters=[("date", "==", ts)])
        df = _post(df)
    else:
        target = ts.strftime("%d-%b-%Y")
        parts = []
        for chunk in pd.read_csv(csv_path(symbol), chunksize=500_000,
                                 low_memory=False):
            sub = chunk[chunk["FH_TIMESTAMP"] == target]
            if len(sub):
                parts.append(_tidy(sub))
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if liquid_only and len(df):
        df = df[df["volume"] > 0].copy()
    return df.reset_index(drop=True)


def _post(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure derived cols exist when loading straight from Parquet."""
    if "T" not in df.columns:
        df["T"] = (df["expiry"] - df["date"]).dt.days / 365.0
    return df


def get_spot(chain: pd.DataFrame) -> float:
    """Underlying value for the chain (single trade date)."""
    s = chain["spot"].dropna()
    s = s[s > 0]
    return float(s.iloc[0]) if len(s) else np.nan


def expiry_slice(chain: pd.DataFrame, expiry: str) -> pd.DataFrame:
    """One expiry's options, sorted by strike."""
    exp = pd.to_datetime(expiry)
    return chain[chain["expiry"] == exp].sort_values("strike").reset_index(drop=True)
