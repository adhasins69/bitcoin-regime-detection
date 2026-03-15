"""
features.py
===========
All feature engineering in one place.  No external TA library required —
every indicator is implemented from scratch using pandas and numpy.

Public API
----------
compute_hmm_features(df)  ->  (X: np.ndarray, index: pd.Index)
    Returns the (n_samples, 3) feature matrix used to train and query
    the Hidden Markov Model, along with the aligned DatetimeIndex.

compute_indicators(df)    ->  pd.DataFrame
    Returns an enriched copy of df with all 12 indicator columns appended.
    Used by the strategy vote system and the Streamlit dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# HMM feature matrix  (3 features — compact, orthogonal signal set)
# ---------------------------------------------------------------------------

def compute_hmm_features(df: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
    """
    Build the (n_samples, 3) feature matrix for the regime model.

    Features
    --------
    0. Returns      – Close.pct_change()            bar-to-bar price direction
    1. Range        – (High - Low) / Close           intra-bar volatility proxy
    2. Vol_Change   – Volume.pct_change().rolling(20).std()
                      rolling std of volume pct-changes, captures liquidity spikes

    Rows containing NaN (warm-up period) are dropped before returning.
    The index is returned so the caller can align regime predictions back
    onto the original DataFrame without index mismatch errors.
    """
    returns    = df["Close"].pct_change()
    rng        = (df["High"] - df["Low"]) / df["Close"]
    vol_change = df["Volume"].pct_change().rolling(20).std()

    feat = pd.DataFrame(
        {"Returns": returns, "Range": rng, "Vol_Change": vol_change},
        index=df.index,
    )

    # Drop NaN (first row of Returns, first 20 rows of Vol_Change rolling window)
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.dropna(inplace=True)

    return feat.values, feat.index


# ---------------------------------------------------------------------------
# Full indicator set  (12 columns appended to OHLCV)
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicator columns and return an enriched copy of df.

    Added columns
    -------------
    Returns        – simple bar-to-bar return  (Close.pct_change)
    Range          – (High - Low) / Close
    Vol_Change     – Volume.pct_change
    RSI            – Wilder RSI, 14-period
    Momentum       – 10-bar price momentum as %
    Volatility     – annualised rolling volatility, 20-bar window, %
    Vol_SMA_20     – 20-bar volume simple moving average
    ADX            – Wilder Average Directional Index, 14-period
    EMA_50         – Exponential Moving Average, 50-period
    EMA_200        – Exponential Moving Average, 200-period
    MACD           – MACD line (EMA12 - EMA26)
    Signal_Line    – MACD signal line (EMA9 of MACD)

    NaN / inf values are replaced after computation so every row that
    has enough history for EMA_200 is usable (no row is silently dropped).
    """
    out    = df.copy()
    close  = out["Close"]
    high   = out["High"]
    low    = out["Low"]
    volume = out["Volume"]

    # ── Core regime features ──────────────────────────────────────────────
    out["Returns"]    = close.pct_change()
    out["Range"]      = (high - low) / close
    out["Vol_Change"] = volume.pct_change()

    # ── Oscillators ───────────────────────────────────────────────────────
    out["RSI"]      = _rsi(close, 14)
    out["Momentum"] = close.pct_change(10) * 100

    log_ret             = np.log(close / close.shift(1))
    out["Volatility"]   = log_ret.rolling(20).std() * np.sqrt(24 * 365) * 100

    out["Vol_SMA_20"]   = volume.rolling(20).mean()

    # ── Trend ─────────────────────────────────────────────────────────────
    out["ADX"]          = _adx(high, low, close, 14)
    out["EMA_50"]       = _ema(close, 50)
    out["EMA_200"]      = _ema(close, 200)
    out["MACD"], out["Signal_Line"] = _macd(close)

    # ── Sanitise: replace inf / -inf with NaN ─────────────────────────────
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    return out


# ---------------------------------------------------------------------------
# Private indicator implementations
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average using pandas ewm (adjust=False = Wilder-style)."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    Uses a simple rolling-mean approximation (Cutler's RSI) which is
    stable and consistent. Values are in [0, 100].
    """
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _adx(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average Directional Index (ADX) — Wilder-smoothed.

    ADX measures trend strength regardless of direction.
    Values > 25 are conventionally interpreted as a trending market.
    """
    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    plus_di  = (
        100.0
        * pd.Series(plus_dm, index=close.index)
        .ewm(alpha=1.0 / period, adjust=False).mean()
        / atr
    )
    minus_di = (
        100.0
        * pd.Series(minus_dm, index=close.index)
        .ewm(alpha=1.0 / period, adjust=False).mean()
        / atr
    )

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx    = 100.0 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def _macd(
    close:  pd.Series,
    fast:   int = 12,
    slow:   int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """
    MACD line and Signal line.

    MACD      = EMA(fast) - EMA(slow)
    Signal    = EMA(signal) of MACD
    """
    macd_line   = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line
