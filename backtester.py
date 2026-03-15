"""
backtester.py
=============
Self-contained event-driven backtesting engine.

Pipeline (orchestrated by BacktestEngine.run)
---------------------------------------------
  raw OHLCV  →  features  →  HMM regime model  →  strategy signals
              →  _simulate (event loop)          →  BacktestResult
              →  compute_metrics                 →  dict

Trading assumptions
-------------------
- Long-only, no leverage, no shorting
- Full capital deployment when entering a trade
- Fee and slippage applied symmetrically on entry and exit
- Entry: on the Close price of the bar where signal turns LONG
- Exit:  on the Close price of the bar where signal turns CASH or
         regime turns Bear (whichever comes first)
- Any open trade at end of data is closed at the final bar's Close

Public API
----------
BacktestEngine(cfg).run(df=None)  ->  BacktestResult
compute_metrics(result, df_raw)   ->  dict
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import AppConfig, DEFAULT_CONFIG
from data_loader import fetch_data
from features import compute_hmm_features, compute_indicators
from regime_model import RegimeModel, BEAR
from strategy import generate_signals, SIGNAL_LONG

warnings.filterwarnings("ignore")

# Output directory for CSV exports (created at import time)
OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """
    Record of a single closed trade.

    entry_price / exit_price are cost-adjusted (slippage + fee included).
    raw_entry_price / raw_exit_price are the unmodified market Close prices.
    pnl and return_pct are computed automatically in __post_init__.
    """
    entry_time:      pd.Timestamp
    exit_time:       pd.Timestamp
    entry_price:     float          # cost-adjusted entry price
    exit_price:      float          # cost-adjusted exit price
    raw_entry_price: float          # raw Close at entry
    raw_exit_price:  float          # raw Close at exit
    entry_capital:   float          # portfolio value at trade open
    exit_capital:    float          # portfolio value at trade close
    exit_reason:     str            # "Signal Off", "Bear Regime", or "End of Data"
    votes_at_entry:  int            # confirmation vote count when entering

    # Computed fields
    pnl:        float = field(init=False)
    return_pct: float = field(init=False)

    def __post_init__(self):
        if self.entry_price > 0:
            self.return_pct = (self.exit_price - self.entry_price) / self.entry_price * 100
        else:
            self.return_pct = 0.0
        self.pnl = self.exit_capital - self.entry_capital


@dataclass
class BacktestResult:
    """Container for all outputs of a single backtest run."""
    trades:       list[Trade]
    equity_curve: pd.DataFrame    # columns: value, regime, in_position
    signal_df:    pd.DataFrame    # full bar-level DataFrame (features + signals)
    config:       AppConfig
    regime_model: Optional[RegimeModel] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Orchestrates the full data → features → regime → signal → simulation pipeline.

    Usage
    -----
    >>> engine = BacktestEngine(cfg)
    >>> result = engine.run()               # fetches data automatically
    >>> result = engine.run(df=my_ohlcv)   # use a pre-fetched DataFrame
    """

    def __init__(self, cfg: AppConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    def run(self, df: Optional[pd.DataFrame] = None) -> BacktestResult:
        """
        Execute the full backtest pipeline.

        Parameters
        ----------
        df : Optional pre-fetched OHLCV DataFrame.
             If None, data is downloaded using cfg.data settings.

        Returns
        -------
        BacktestResult
        """
        cfg = self.cfg

        # ── Stage 1: Data ─────────────────────────────────────────────────
        if df is None:
            df = fetch_data(cfg.data)

        # ── Stage 2: Feature engineering ──────────────────────────────────
        ind_df      = compute_indicators(df)
        X, feat_idx = compute_hmm_features(df)

        # ── Stage 3: Train HMM regime model ───────────────────────────────
        model   = RegimeModel(cfg.hmm)
        model.fit(X)
        regimes = model.predict_series(X, feat_idx)

        # ── Stage 4: Generate strategy signals ────────────────────────────
        signal_df = generate_signals(ind_df, regimes, cfg.strategy)

        # ── Stage 5: Simulate portfolio ───────────────────────────────────
        trades, equity_curve = self._simulate(signal_df, cfg)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            signal_df=signal_df,
            config=cfg,
            regime_model=model,
        )

    def _simulate(
        self,
        signal_df: pd.DataFrame,
        cfg:       AppConfig,
    ) -> tuple[list[Trade], pd.DataFrame]:
        """
        Core event loop — iterate bar-by-bar and manage positions.

        Cost model
        ----------
        For a trade at raw market price P:
            entry_price = P * (1 + fee + slippage)   # you pay more on entry
            exit_price  = P * (1 - fee - slippage)   # you receive less on exit

        Return on capital for a trade:
            r = (exit_price - entry_price) / entry_price
            capital_after = capital_before * (1 + r)
        """
        bt   = cfg.backtest
        cost = (bt.fee_pct + bt.slippage_pct) / 100.0  # fractional cost per side

        capital     = bt.starting_capital
        pos: Optional[dict] = None     # open position state dict
        trades:     list[Trade] = []
        equity_rows: list[dict] = []

        for i in range(len(signal_df)):
            row       = signal_df.iloc[i]
            timestamp = signal_df.index[i]
            raw_price = float(row["Close"])
            regime    = str(row["regime"])
            signal    = str(row["signal"])

            # ── Mark-to-market ─────────────────────────────────────────────
            if pos is not None:
                # Current value if we were to close right now
                adj_exit   = raw_price * (1.0 - cost)
                cur_return = (adj_exit - pos["adj_entry"]) / pos["adj_entry"]
                cur_val    = pos["entry_capital"] * (1.0 + cur_return)
            else:
                cur_val = capital

            equity_rows.append({
                "time":        timestamp,
                "value":       max(cur_val, 0.0),
                "regime":      regime,
                "in_position": pos is not None,
            })

            # ── Exit check (if we hold a position) ─────────────────────────
            if pos is not None:
                # Exit when: signal flips to CASH, OR regime turns Bear
                should_exit  = (signal != SIGNAL_LONG) or (regime == BEAR)
                exit_reason  = (
                    "Bear Regime" if regime == BEAR
                    else "Signal Off"
                )

                if should_exit:
                    adj_exit    = raw_price * (1.0 - cost)
                    cur_return  = (adj_exit - pos["adj_entry"]) / pos["adj_entry"]
                    exit_cap    = pos["entry_capital"] * (1.0 + cur_return)

                    trades.append(Trade(
                        entry_time      = pos["entry_time"],
                        exit_time       = timestamp,
                        entry_price     = pos["adj_entry"],
                        exit_price      = adj_exit,
                        raw_entry_price = pos["raw_entry"],
                        raw_exit_price  = raw_price,
                        entry_capital   = pos["entry_capital"],
                        exit_capital    = exit_cap,
                        exit_reason     = exit_reason,
                        votes_at_entry  = pos["votes"],
                    ))

                    capital = exit_cap
                    pos     = None

                continue  # skip entry check on exit bar (or while still in trade)

            # ── Entry check (if flat) ──────────────────────────────────────
            if signal == SIGNAL_LONG:
                adj_entry = raw_price * (1.0 + cost)
                pos = {
                    "raw_entry":     raw_price,
                    "adj_entry":     adj_entry,
                    "entry_time":    timestamp,
                    "entry_capital": capital,
                    "votes":         int(row["votes"]),
                }

        # ── Force-close any open position at end of data ───────────────────
        if pos is not None:
            raw_price  = float(signal_df["Close"].iloc[-1])
            timestamp  = signal_df.index[-1]
            adj_exit   = raw_price * (1.0 - cost)
            cur_return = (adj_exit - pos["adj_entry"]) / pos["adj_entry"]
            exit_cap   = pos["entry_capital"] * (1.0 + cur_return)

            trades.append(Trade(
                entry_time      = pos["entry_time"],
                exit_time       = timestamp,
                entry_price     = pos["adj_entry"],
                exit_price      = adj_exit,
                raw_entry_price = pos["raw_entry"],
                raw_exit_price  = raw_price,
                entry_capital   = pos["entry_capital"],
                exit_capital    = exit_cap,
                exit_reason     = "End of Data",
                votes_at_entry  = pos["votes"],
            ))

        equity_df = pd.DataFrame(equity_rows).set_index("time")
        return trades, equity_df


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(result: BacktestResult, df_raw: pd.DataFrame) -> dict:
    """
    Compute a comprehensive performance summary from a BacktestResult.

    Parameters
    ----------
    result : BacktestResult from BacktestEngine.run()
    df_raw : The raw OHLCV DataFrame used in the backtest (for buy-and-hold)

    Returns
    -------
    dict with keys:
        total_return_pct, bh_return_pct, alpha,
        win_rate, max_drawdown, sharpe,
        num_trades, final_value, start_capital,
        current_signal, current_regime, current_votes,
        mode, min_votes
    """
    equity  = result.equity_curve
    trades  = result.trades
    cfg     = result.config

    start_cap   = cfg.backtest.starting_capital
    final_value = float(equity["value"].iloc[-1])
    total_ret   = (final_value / start_cap - 1.0) * 100.0

    # ── Buy-and-hold benchmark ─────────────────────────────────────────────
    bh_prices = df_raw["Close"].reindex(equity.index, method="ffill")
    bh_ret    = float((bh_prices.iloc[-1] / bh_prices.iloc[0] - 1.0) * 100.0)
    alpha     = total_ret - bh_ret

    # ── Win rate ───────────────────────────────────────────────────────────
    wins     = sum(1 for t in trades if t.pnl > 0)
    win_rate = (wins / len(trades) * 100.0) if trades else 0.0

    # ── Maximum drawdown ───────────────────────────────────────────────────
    running_max  = equity["value"].cummax()
    drawdown_ser = (equity["value"] - running_max) / running_max * 100.0
    max_dd       = float(drawdown_ser.min())

    # ── Annualised Sharpe ratio (hourly bars, 24×365 trading hours/year) ───
    port_rets = equity["value"].pct_change().dropna()
    if port_rets.std() > 1e-12:
        sharpe = float(port_rets.mean() / port_rets.std() * np.sqrt(24 * 365))
    else:
        sharpe = 0.0

    # ── Current state (latest bar) ─────────────────────────────────────────
    last_row    = result.signal_df.iloc[-1]
    cur_regime  = str(last_row["regime"])
    cur_votes   = int(last_row["votes"])
    in_pos      = bool(equity["in_position"].iloc[-1])
    min_v       = cfg.strategy.min_votes

    if in_pos:
        cur_signal = SIGNAL_LONG
    elif cur_regime == "Bull" and cur_votes >= min_v:
        cur_signal = "LONG (Entry Pending)"
    else:
        cur_signal = "CASH"

    return {
        "total_return_pct": round(total_ret,   2),
        "bh_return_pct":    round(bh_ret,      2),
        "alpha":            round(alpha,        2),
        "win_rate":         round(win_rate,     2),
        "max_drawdown":     round(max_dd,       2),
        "sharpe":           round(sharpe,       3),
        "num_trades":       len(trades),
        "final_value":      round(final_value,  2),
        "start_capital":    start_cap,
        "current_signal":   cur_signal,
        "current_regime":   cur_regime,
        "current_votes":    cur_votes,
        "mode":             cfg.strategy.mode,
        "min_votes":        min_v,
    }


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Convert a list of Trade objects into a display-ready DataFrame."""
    if not trades:
        return pd.DataFrame()

    return pd.DataFrame([{
        "entry_time":      t.entry_time,
        "exit_time":       t.exit_time,
        "entry_price":     round(t.raw_entry_price, 2),
        "exit_price":      round(t.raw_exit_price, 2),
        "entry_price_adj": round(t.entry_price, 2),
        "exit_price_adj":  round(t.exit_price, 2),
        "entry_capital":   round(t.entry_capital, 2),
        "exit_capital":    round(t.exit_capital, 2),
        "pnl":             round(t.pnl, 2),
        "return_pct":      round(t.return_pct, 4),
        "exit_reason":     t.exit_reason,
        "votes_at_entry":  t.votes_at_entry,
    } for t in trades])
