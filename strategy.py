"""
strategy.py
===========
Converts regime labels and indicator values into LONG / CASH signals
using an 8-condition confirmation vote system.

Entry rule
----------
LONG  ← regime == "Bull" AND confirmation_votes >= strategy_cfg.min_votes
CASH  ← otherwise

The confirmation system prevents the strategy from entering on regime
alone — it requires technical alignment across multiple dimensions:
momentum, volatility, volume, trend, and oscillator state.

Public API
----------
count_votes(row, cfg)              -> int        [0, 8]
generate_signals(ind_df, regimes, cfg) -> pd.DataFrame
"""

from __future__ import annotations

import pandas as pd

from config import StrategyConfig, NORMAL_MODE
from regime_model import BULL

# Signal label constants — used by backtester and dashboard
SIGNAL_LONG = "LONG"
SIGNAL_CASH = "CASH"


# ---------------------------------------------------------------------------
# Vote counter  (one bar → one integer)
# ---------------------------------------------------------------------------

def count_votes(row: pd.Series, cfg: StrategyConfig = NORMAL_MODE) -> int:
    """
    Evaluate the 8 confirmation conditions for a single bar.

    Conditions
    ----------
    1.  RSI < 90               – not overbought
    2.  Momentum > 1 %         – positive short-term price momentum
    3.  Volatility < 6 %       – annualised vol is manageable
    4.  Volume > Vol_SMA_20    – above-average participation
    5.  ADX > 25               – trend is strong enough to trade
    6.  Close > EMA_50         – short-term uptrend alignment
    7.  Close > EMA_200        – long-term uptrend alignment
    8.  MACD > Signal_Line     – momentum is accelerating upward

    Thresholds for conditions 1-3 and 5 are driven by cfg so they can be
    adjusted without editing this function.

    Returns
    -------
    int in [0, 8]
    """
    score = 0

    # Condition 1 — RSI not overbought
    score += int(row.get("RSI",         100.0) < cfg.rsi_max)

    # Condition 2 — Positive momentum
    score += int(row.get("Momentum",      0.0) > cfg.momentum_min)

    # Condition 3 — Volatility within bounds
    score += int(row.get("Volatility",  100.0) < cfg.vol_max)

    # Condition 4 — Volume above 20-bar SMA
    score += int(row.get("Volume", 0.0) > row.get("Vol_SMA_20", float("inf")))

    # Condition 5 — Trend is strong
    score += int(row.get("ADX", 0.0) > cfg.adx_min)

    # Condition 6 — Price above EMA 50
    score += int(row.get("Close", 0.0) > row.get("EMA_50", float("inf")))

    # Condition 7 — Price above EMA 200
    score += int(row.get("Close", 0.0) > row.get("EMA_200", float("inf")))

    # Condition 8 — MACD above signal line
    score += int(row.get("MACD", 0.0) > row.get("Signal_Line", 0.0))

    return score


# ---------------------------------------------------------------------------
# Signal generation  (full DataFrame → annotated DataFrame)
# ---------------------------------------------------------------------------

def generate_signals(
    ind_df:  pd.DataFrame,
    regimes: pd.Series,
    cfg:     StrategyConfig = NORMAL_MODE,
) -> pd.DataFrame:
    """
    Attach regime, votes, signal, and position columns to a copy of ind_df.

    Parameters
    ----------
    ind_df  : pd.DataFrame
        Output of features.compute_indicators() — contains OHLCV + all indicator
        columns.
    regimes : pd.Series
        Output of RegimeModel.predict_series() — index is the HMM-aligned
        DatetimeIndex (slightly shorter than ind_df due to warm-up rows).
    cfg     : StrategyConfig
        Controls which mode (Normal / Aggressive) and vote threshold to use.

    Returns
    -------
    pd.DataFrame
        Aligned to the regime index.  Added columns:
        - regime  : "Bull" / "Bear" / "Neutral"
        - votes   : int [0, 8] — number of confirmations passed
        - signal  : "LONG" or "CASH"
        - position: 1 (long) or 0 (cash) — forward-filled for equity curve
    """
    # Align to the regime index (HMM warm-up rows are dropped here)
    out = ind_df.loc[regimes.index].copy()

    # Drop rows where key indicators haven't warmed up yet
    out.dropna(subset=["RSI", "ADX", "EMA_200"], inplace=True)

    aligned_regimes = regimes.loc[out.index]
    out["regime"] = aligned_regimes

    # Vectorised vote count using row-wise apply
    out["votes"] = out.apply(lambda r: count_votes(r, cfg), axis=1)

    # Entry signal: Bull regime AND enough confirmations
    out["signal"] = out.apply(
        lambda r: SIGNAL_LONG
        if (r["regime"] == BULL and r["votes"] >= cfg.min_votes)
        else SIGNAL_CASH,
        axis=1,
    )

    # Position column: 1 = long, 0 = cash
    out["position"] = (out["signal"] == SIGNAL_LONG).astype(int)

    return out
