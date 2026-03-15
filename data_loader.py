"""
data_loader.py
==============
Downloads and cleans hourly BTC-USD OHLCV data from yfinance.

Public API
----------
fetch_data(cfg) -> pd.DataFrame
    Download BTC-USD hourly data for the period defined in DataConfig.
    Returns a clean DataFrame with columns [Open, High, Low, Close, Volume]
    and a tz-aware UTC DatetimeIndex, sorted ascending.
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
import yfinance as yf

from config import DataConfig, DEFAULT_CONFIG

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_data(cfg: DataConfig = DEFAULT_CONFIG.data) -> pd.DataFrame:
    """
    Download BTC-USD hourly OHLCV data using yfinance.

    Uses period= and interval= directly (yfinance native), which is more
    reliable than explicit start/end dates for hourly granularity.

    Parameters
    ----------
    cfg : DataConfig
        Ticker, period string, and interval.

    Returns
    -------
    pd.DataFrame
        Clean OHLCV DataFrame with a tz-aware UTC DatetimeIndex.
        Columns: Open, High, Low, Close, Volume.
        Sorted ascending, duplicates removed, zero-volume rows dropped.

    Raises
    ------
    RuntimeError
        If yfinance returns empty or unusably small data.
    """
    raw = _download(cfg.ticker, cfg.period, cfg.interval)

    if raw is None or raw.empty or len(raw) < 300:
        raise RuntimeError(
            f"yfinance returned too little data for {cfg.ticker} "
            f"(period={cfg.period}, interval={cfg.interval}). "
            "Check your internet connection or try again."
        )

    return _clean(raw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    """Thin wrapper around yfinance.download with error handling."""
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        return raw if (raw is not None and not raw.empty) else None
    except Exception as exc:
        raise RuntimeError(
            f"yfinance download failed for {ticker}: {exc}"
        ) from exc


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise a raw yfinance DataFrame into a clean OHLCV frame.

    Steps
    -----
    1. Flatten MultiIndex columns produced by newer yfinance versions.
    2. Normalise column names to Title Case.
    3. Rename any 'Adj Close' variant to 'Close'.
    4. Keep only [Open, High, Low, Close, Volume].
    5. Coerce to numeric, drop NaN rows.
    6. Drop zero-volume rows (market-closed artefacts).
    7. Ensure the index is a tz-aware UTC DatetimeIndex.
    8. Sort ascending and remove duplicate timestamps.
    """
    # ── 1. Flatten MultiIndex columns (yfinance >= 0.2.38 quirk) ──────────
    if isinstance(df.columns, pd.MultiIndex):
        # Take the first level (price type), drop ticker level
        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    # ── 2. Normalise to Title Case  ("open" → "Open") ─────────────────────
    df.columns = [str(c).strip().title() for c in df.columns]

    # ── 3. Rename 'Adj Close' → 'Close' ────────────────────────────────────
    rename_map = {
        c: "Close"
        for c in df.columns
        if "Adj" in c and "Close" in c
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    # ── 4. Keep only the five standard OHLCV columns ───────────────────────
    needed = ["Open", "High", "Low", "Close", "Volume"]
    present = [c for c in needed if c in df.columns]
    if len(present) < 5:
        raise RuntimeError(
            f"Expected columns {needed}, but yfinance only returned {list(df.columns)}. "
            "Try updating yfinance: pip install --upgrade yfinance"
        )
    df = df[needed].copy()

    # ── 5. Enforce numeric dtypes; drop NaN rows ───────────────────────────
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)

    # ── 6. Drop zero-volume rows (weekends / market-close artefacts) ───────
    df = df[df["Volume"] > 0].copy()

    # ── 7. Ensure tz-aware UTC index (Windows-safe) ────────────────────────
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # ── 8. Sort and deduplicate ────────────────────────────────────────────
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    return df
