"""
paper_trader.py
===============
Paper-trading module.  Fetches live market data, generates signals via the
full HMM + indicator pipeline, executes simulated orders through PaperBroker,
and logs every tick and trade to CSV files.

No real money is ever touched.

Public API
----------
PaperTrader(cfg, broker).run_once()  →  status dict
PaperTrader.get_status()             →  status dict
PaperTrader.get_trade_log()          →  pd.DataFrame  (from CSV)
PaperTrader.get_tick_log()           →  pd.DataFrame  (from CSV)
PaperTrader.reset()                  →  wipes state + CSV logs
"""

from __future__ import annotations

import csv
import warnings
from datetime import timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from config import AppConfig, DEFAULT_CONFIG
from data_loader import fetch_latest
from features import compute_hmm_features, compute_indicators
from regime_model import RegimeModel
from strategy import generate_signals, SIGNAL_LONG
from risk_manager import RiskManager
from broker import PaperBroker

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CSV file paths  (relative to the working directory)
# ---------------------------------------------------------------------------

TRADE_LOG_FILE = Path("paper_trades.csv")
TICK_LOG_FILE  = Path("paper_ticks.csv")

_TRADE_HEADERS = [
    "timestamp", "side", "raw_price", "adj_price",
    "units", "capital_before", "capital_after",
    "pnl", "exit_reason", "votes", "regime", "mode",
]
_TICK_HEADERS = [
    "timestamp", "close", "regime", "votes", "signal",
    "balance", "in_position", "position_entry_price",
]


# ---------------------------------------------------------------------------
# PaperTrader
# ---------------------------------------------------------------------------

class PaperTrader:
    """
    Parameters
    ----------
    cfg    : AppConfig  — data, HMM, risk, and mode settings
    broker : PaperBroker  — provide your own or a default is created
    """

    def __init__(
        self,
        cfg:    AppConfig         = DEFAULT_CONFIG,
        broker: Optional[PaperBroker] = None,
    ):
        self.cfg    = cfg
        self.broker = broker or PaperBroker(
            starting_capital=cfg.risk.starting_capital,
            persist=True,
        )
        self._rm    = RiskManager(cfg.risk, cfg.mode)
        self._model: Optional[RegimeModel] = None

        # In-memory state that mirrors what we'd need from the broker
        self._peak_price:   float = 0.0
        self._last_signal:  str   = "CASH"
        self._last_regime:  str   = "Unknown"
        self._last_votes:   int   = 0
        self._last_price:   float = 0.0

        self._init_logs()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_once(self) -> dict:
        """
        1. Fetch the latest market data.
        2. Compute features + regime + signals.
        3. Act on the signal via the paper broker.
        4. Log tick + trade.
        5. Return current status dict.
        """
        # ── Fetch & compute ────────────────────────────────────────────
        try:
            df = fetch_latest(self.cfg.data)
        except Exception as e:
            return {**self._status(), "error": str(e)}

        ind_df      = compute_indicators(df)
        X, feat_idx = compute_hmm_features(df)

        if len(feat_idx) < self.cfg.hmm.n_components * 3:
            return {**self._status(), "error": "Not enough data to fit HMM."}

        # Re-fit on the fresh window every run (keeps regime labels current)
        self._model = RegimeModel(self.cfg.hmm)
        self._model.fit(X)
        regimes   = self._model.predict_series(X, feat_idx)
        signal_df = generate_signals(ind_df, regimes, self.cfg.mode)

        if signal_df.empty:
            return {**self._status(), "error": "Signal DataFrame is empty."}

        latest    = signal_df.iloc[-1]
        raw_price = float(latest["Close"])
        ts        = str(signal_df.index[-1])

        self._last_signal = str(latest["signal"])
        self._last_regime = str(latest["regime"])
        self._last_votes  = int(latest["votes"])
        self._last_price  = raw_price

        # ── Log tick ───────────────────────────────────────────────────
        pos = self.broker.get_position()
        self._log_tick(ts, latest, pos)

        # ── Execute signal ─────────────────────────────────────────────
        current_time = signal_df.index[-1]

        if self._last_signal == SIGNAL_LONG and pos is None:
            # Entry: only if not in cooldown
            last_exit = self._last_exit_time()
            if not self._rm.in_cooldown(last_exit, current_time):
                self._open_long(raw_price, ts, latest)

        elif pos is not None:
            # Exit check
            self._peak_price = max(self._peak_price, raw_price)
            do_exit, reason  = self._rm.should_exit(
                self._last_regime, raw_price, self._peak_price
            )
            if do_exit:
                self._close_long(raw_price, ts, reason, latest)

        return self._status()

    # ------------------------------------------------------------------
    # Status & log readers
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return self._status()

    def get_trade_log(self) -> pd.DataFrame:
        """Return the paper trade log as a DataFrame (empty if no trades yet)."""
        if not TRADE_LOG_FILE.exists():
            return pd.DataFrame(columns=_TRADE_HEADERS)
        try:
            df = pd.read_csv(TRADE_LOG_FILE)
            return df
        except Exception:
            return pd.DataFrame(columns=_TRADE_HEADERS)

    def get_tick_log(self, last_n: int = 100) -> pd.DataFrame:
        """Return the last N ticks as a DataFrame."""
        if not TICK_LOG_FILE.exists():
            return pd.DataFrame(columns=_TICK_HEADERS)
        try:
            df = pd.read_csv(TICK_LOG_FILE)
            return df.tail(last_n)
        except Exception:
            return pd.DataFrame(columns=_TICK_HEADERS)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Wipe broker state and CSV logs. Starts fresh."""
        self.broker.reset(self.cfg.risk.starting_capital)
        self._peak_price  = 0.0
        self._last_signal = "CASH"
        self._last_regime = "Unknown"
        self._last_votes  = 0
        self._last_price  = 0.0
        TRADE_LOG_FILE.unlink(missing_ok=True)
        TICK_LOG_FILE.unlink(missing_ok=True)
        self._init_logs()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_long(self, raw_price: float, ts: str, row: pd.Series):
        adj_price      = self._rm.adjusted_entry_price(raw_price)
        capital_before = self.broker.get_balance()
        alloc          = self._rm.position_size(capital_before)
        units          = alloc / adj_price

        try:
            self.broker.place_market_buy(adj_price, units)
            self._peak_price = raw_price
            capital_after    = self.broker.get_balance()
            self._log_trade(
                ts=ts, side="BUY",
                raw_price=raw_price, adj_price=adj_price, units=units,
                capital_before=capital_before, capital_after=capital_after,
                pnl=0.0, exit_reason="—",
                votes=int(row["votes"]), regime=str(row["regime"]),
            )
        except ValueError:
            pass  # Insufficient balance — skip

    def _close_long(
        self, raw_price: float, ts: str, reason: str, row: pd.Series
    ):
        adj_price      = self._rm.adjusted_exit_price(raw_price)
        capital_before = self.broker.get_balance()
        result         = self.broker.close_position(adj_price)
        capital_after  = self.broker.get_balance()
        pnl            = float(result.get("pnl", 0.0))
        units          = float(result.get("units", 0.0))
        self._peak_price = 0.0
        self._log_trade(
            ts=ts, side="SELL",
            raw_price=raw_price, adj_price=adj_price, units=units,
            capital_before=capital_before, capital_after=capital_after,
            pnl=pnl, exit_reason=reason,
            votes=int(row["votes"]), regime=str(row["regime"]),
        )

    def _last_exit_time(self) -> Optional[pd.Timestamp]:
        """Read the most recent SELL timestamp from the trade log."""
        if not TRADE_LOG_FILE.exists():
            return None
        try:
            rows  = list(csv.DictReader(TRADE_LOG_FILE.open()))
            sells = [r for r in rows if r.get("side") == "SELL"]
            if sells:
                ts = pd.Timestamp(sells[-1]["timestamp"])
                # Ensure tz-aware
                if ts.tz is None:
                    ts = ts.tz_localize("UTC")
                return ts
        except Exception:
            pass
        return None

    def _status(self) -> dict:
        pos = self.broker.get_position()
        return {
            "signal":           self._last_signal,
            "regime":           self._last_regime,
            "votes":            self._last_votes,
            "last_price":       self._last_price,
            "balance":          self.broker.get_balance(),
            "in_position":      pos is not None,
            "position":         pos,
            "mode":             self.cfg.mode.name,
            "leverage":         self.cfg.mode.leverage,
            "trailing_stop":    self.cfg.mode.trailing_stop_pct,
        }

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------

    def _init_logs(self):
        for fpath, headers in [
            (TRADE_LOG_FILE, _TRADE_HEADERS),
            (TICK_LOG_FILE,  _TICK_HEADERS),
        ]:
            if not fpath.exists():
                with fpath.open("w", newline="") as f:
                    csv.writer(f).writerow(headers)

    def _log_trade(
        self, ts, side, raw_price, adj_price, units,
        capital_before, capital_after, pnl, exit_reason, votes, regime,
    ):
        with TRADE_LOG_FILE.open("a", newline="") as f:
            csv.writer(f).writerow([
                ts,
                side,
                f"{raw_price:.2f}",
                f"{adj_price:.2f}",
                f"{units:.6f}",
                f"{capital_before:.2f}",
                f"{capital_after:.2f}",
                f"{pnl:.2f}",
                exit_reason,
                votes,
                regime,
                self.cfg.mode.name,
            ])

    def _log_tick(self, ts: str, row: pd.Series, pos: Optional[dict]):
        entry_px = f"{pos['entry_price']:.2f}" if pos else "—"
        with TICK_LOG_FILE.open("a", newline="") as f:
            csv.writer(f).writerow([
                ts,
                f"{row['Close']:.2f}",
                row["regime"],
                int(row["votes"]),
                row["signal"],
                f"{self.broker.get_balance():.2f}",
                pos is not None,
                entry_px,
            ])
