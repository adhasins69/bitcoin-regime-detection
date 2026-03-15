"""
live_trader.py
==============
Autonomous Binance Spot live trading loop.

Run this script directly — it loops forever, trading BTC/USDT on Binance Spot
using the HMM regime model and 8-condition vote system.

Safety guarantees
-----------------
• LIVE_TRADING_ENABLED must be set to true in .env (default is False in config)
• LIVE_CONFIRM env var must equal "YES_LIVE_TRADING" exactly
• Only one open position at a time — never pyramids
• Max order size: 50 USD notional (configurable in LiveConfig)
• 48-hour cooldown after every exit
• 60-minute minimum gap between any two trades
• Every failed order is logged and the cycle continues — no blind retry
• HMM is retrained on each cycle from the latest Binance klines

Usage (PowerShell)
------------------
    python live_trader.py

Logs
----
  live_cycles.csv      — one row per loop cycle (signal, balance, action, …)
  live_trades.jsonl    — one JSON line per executed trade
  live_state.json      — persisted state (entry price, cooldown timer, …)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env before importing config (so env vars override defaults)
load_dotenv()

from config import DEFAULT_CONFIG, LiveConfig, HMMConfig
from data_loader import fetch_binance_klines
from features import compute_hmm_features, compute_indicators
from regime_model import RegimeModel, BULL
from strategy import generate_signals, SIGNAL_LONG, SIGNAL_CASH, NORMAL_MODE
from risk_manager import LiveRiskManager
from broker import BinanceSpotBroker


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("live_trader.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("live_trader")


# ---------------------------------------------------------------------------
# Live state persistence
# ---------------------------------------------------------------------------

class LiveState:
    """
    Persists critical state to disk so the trader survives restarts.

    Stored fields
    -------------
    entry_price     : float | None  — USD price at entry
    entry_time      : str   | None  — ISO-8601 UTC timestamp
    entry_units     : float | None  — BTC bought
    last_exit_time  : str   | None  — ISO-8601 UTC timestamp of last close
    last_trade_time : str   | None  — ISO-8601 UTC timestamp of last trade
    cycle_count     : int           — total cycles completed
    """

    def __init__(self, path: str = "live_state.json"):
        self._path = Path(path)
        self.entry_price:     Optional[float] = None
        self.entry_time:      Optional[str]   = None
        self.entry_units:     Optional[float] = None
        self.last_exit_time:  Optional[str]   = None
        self.last_trade_time: Optional[str]   = None
        self.cycle_count:     int             = 0
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.entry_price     = data.get("entry_price")
            self.entry_time      = data.get("entry_time")
            self.entry_units     = data.get("entry_units")
            self.last_exit_time  = data.get("last_exit_time")
            self.last_trade_time = data.get("last_trade_time")
            self.cycle_count     = int(data.get("cycle_count", 0))
        except Exception as exc:
            log.warning(f"Could not load live state from {self._path}: {exc}")

    def save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(
                    {
                        "entry_price":     self.entry_price,
                        "entry_time":      self.entry_time,
                        "entry_units":     self.entry_units,
                        "last_exit_time":  self.last_exit_time,
                        "last_trade_time": self.last_trade_time,
                        "cycle_count":     self.cycle_count,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            log.error(f"Failed to save live state: {exc}")

    def record_entry(self, price: float, units: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.entry_price     = price
        self.entry_time      = now
        self.entry_units     = units
        self.last_trade_time = now
        self.save()

    def record_exit(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.entry_price     = None
        self.entry_time      = None
        self.entry_units     = None
        self.last_exit_time  = now
        self.last_trade_time = now
        self.save()

    @property
    def last_exit_dt(self) -> Optional[datetime]:
        if self.last_exit_time is None:
            return None
        return datetime.fromisoformat(self.last_exit_time)

    @property
    def last_trade_dt(self) -> Optional[datetime]:
        if self.last_trade_time is None:
            return None
        return datetime.fromisoformat(self.last_trade_time)


# ---------------------------------------------------------------------------
# Cycle logger
# ---------------------------------------------------------------------------

class CycleLogger:
    """Writes one CSV row per loop cycle and one JSONL line per trade."""

    CYCLE_FIELDS = [
        "timestamp", "cycle", "signal", "regime", "votes",
        "btc_price", "usdt_balance", "btc_balance", "has_position",
        "action", "in_cooldown", "cooldown_remaining_h",
        "order_id", "executed_qty", "executed_notional",
        "entry_price", "unrealised_pnl_pct", "error",
    ]

    def __init__(self, cycle_file: str = "live_cycles.csv", trade_file: str = "live_trades.jsonl"):
        self._cycle_path = Path(cycle_file)
        self._trade_path = Path(trade_file)

        # Write header if cycle file is new
        if not self._cycle_path.exists():
            with open(self._cycle_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.CYCLE_FIELDS).writeheader()

    def log_cycle(self, data: dict) -> None:
        """Append one row to the cycle CSV."""
        row = {k: data.get(k, "") for k in self.CYCLE_FIELDS}
        try:
            with open(self._cycle_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CYCLE_FIELDS)
                writer.writerow(row)
        except Exception as exc:
            log.error(f"Failed to write cycle log: {exc}")

    def log_trade(self, data: dict) -> None:
        """Append one JSON line to the trade JSONL file."""
        try:
            with open(self._trade_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as exc:
            log.error(f"Failed to write trade log: {exc}")


# ---------------------------------------------------------------------------
# Live trader
# ---------------------------------------------------------------------------

class LiveTrader:
    """
    Autonomous live trading loop for Binance Spot BTCUSDT.

    Architecture
    ------------
    run()        — starts the infinite loop (call this to start the bot)
    run_once()   — executes a single cycle (fetch → signal → act → log)
    _signal()    — runs HMM + indicator pipeline, returns (signal, regime, votes, price)
    _execute()   — places real Binance orders with full pre-flight checks
    """

    def __init__(self):
        self.cfg = DEFAULT_CONFIG.live
        self.risk = LiveRiskManager(self.cfg)

        # Validate environment
        self._api_key    = os.environ.get("BINANCE_API_KEY", "")
        self._api_secret = os.environ.get("BINANCE_API_SECRET", "")
        self._live_confirm = os.environ.get("LIVE_CONFIRM", "")

        # Override LiveConfig from env vars if present
        self._apply_env_overrides()

        # Initialise broker (validates keys are non-empty)
        self.broker = BinanceSpotBroker(
            api_key=self._api_key,
            api_secret=self._api_secret,
            symbol=self.cfg.symbol,
            max_notional_usd=self.cfg.max_notional_usd,
            recv_window=self.cfg.recv_window,
            timeout=self.cfg.request_timeout,
            dust_btc=self.cfg.dust_btc_threshold,
            testnet=False,
        )

        self.state  = LiveState(self.cfg.state_file)
        self.logger = CycleLogger(self.cfg.cycle_log_file, self.cfg.log_file)

        log.info("=" * 60)
        log.info("Live Trader initialised")
        log.info(f"  Symbol          : {self.cfg.symbol}")
        log.info(f"  Max notional    : ${self.cfg.max_notional_usd:.2f} USD")
        log.info(f"  Cooldown        : {self.cfg.cooldown_hours} h")
        log.info(f"  Loop interval   : {self.cfg.loop_interval_seconds} s")
        log.info(f"  Live enabled    : {self.cfg.live_trading_enabled}")
        log.info("=" * 60)

    def _apply_env_overrides(self) -> None:
        """Override LiveConfig fields from environment variables."""
        enabled_str = os.environ.get("LIVE_TRADING_ENABLED", "").lower()
        if enabled_str in ("true", "1", "yes"):
            self.cfg.live_trading_enabled = True
        elif enabled_str in ("false", "0", "no"):
            self.cfg.live_trading_enabled = False

        max_notional = os.environ.get("MAX_LIVE_NOTIONAL_USD")
        if max_notional:
            try:
                self.cfg.max_notional_usd = min(float(max_notional), 50.0)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the infinite trading loop.

        Each iteration:
          1. Run one full cycle (signal → execute → log)
          2. Sleep for loop_interval_seconds
          3. Repeat

        Ctrl-C or any unhandled OS signal will terminate cleanly.
        """
        log.info("Starting live trading loop. Press Ctrl+C to stop.")

        while True:
            cycle_start = time.monotonic()
            self.state.cycle_count += 1
            cycle_num = self.state.cycle_count

            log.info(f"--- Cycle {cycle_num} ---")
            try:
                self.run_once()
            except KeyboardInterrupt:
                log.info("KeyboardInterrupt received. Shutting down gracefully.")
                self.state.save()
                sys.exit(0)
            except Exception as exc:
                log.error(f"Unhandled exception in cycle {cycle_num}: {exc}")
                log.error(traceback.format_exc())
                # Log the error to the cycle log
                self.logger.log_cycle({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cycle":     cycle_num,
                    "action":    "ERROR",
                    "error":     str(exc)[:200],
                })

            elapsed = time.monotonic() - cycle_start
            sleep_s = max(0, self.cfg.loop_interval_seconds - elapsed)
            log.info(f"Cycle {cycle_num} done in {elapsed:.1f}s. Sleeping {sleep_s:.0f}s.")
            time.sleep(sleep_s)

    # ------------------------------------------------------------------
    # Single cycle
    # ------------------------------------------------------------------

    def run_once(self) -> dict:
        """
        Execute one full trading cycle.

        Returns a dict summarising the cycle (same keys as cycle log).
        """
        ts = datetime.now(timezone.utc).isoformat()
        cycle_data: dict = {"timestamp": ts, "cycle": self.state.cycle_count}

        # ── 1. Check kill switch ─────────────────────────────────────────
        try:
            self.risk.assert_live_enabled(self._live_confirm)
        except RuntimeError as exc:
            log.warning(str(exc))
            cycle_data.update({"action": "DISABLED", "error": str(exc)[:200]})
            self.logger.log_cycle(cycle_data)
            return cycle_data

        # ── 2. Fetch Binance klines and run signal pipeline ──────────────
        log.info("Fetching latest Binance klines …")
        try:
            signal, regime, votes, btc_price = self._compute_signal()
            cycle_data.update({
                "signal":    signal,
                "regime":    regime,
                "votes":     votes,
                "btc_price": round(btc_price, 2),
            })
            log.info(
                f"Signal={signal}  Regime={regime}  Votes={votes}/8  "
                f"Price=${btc_price:,.2f}"
            )
        except Exception as exc:
            log.error(f"Signal pipeline failed: {exc}")
            cycle_data.update({"action": "SIGNAL_ERROR", "error": str(exc)[:200]})
            self.logger.log_cycle(cycle_data)
            return cycle_data

        # ── 3. Fetch Binance account state ───────────────────────────────
        log.info("Fetching Binance account state …")
        try:
            snap = self.broker.get_account_snapshot()
            usdt_balance  = snap["usdt_free"]
            btc_balance   = snap["btc_free"]
            has_position  = snap["has_position"]
            cycle_data.update({
                "usdt_balance": round(usdt_balance, 4),
                "btc_balance":  round(btc_balance, 8),
                "has_position": has_position,
            })
            log.info(
                f"Account: USDT=${usdt_balance:.2f}  "
                f"BTC={btc_balance:.6f}  "
                f"HasPosition={has_position}"
            )
        except Exception as exc:
            log.error(f"Account fetch failed: {exc}")
            cycle_data.update({"action": "ACCOUNT_ERROR", "error": str(exc)[:200]})
            self.logger.log_cycle(cycle_data)
            return cycle_data

        # ── 4. Cooldown check ────────────────────────────────────────────
        in_cd = self.risk.is_in_cooldown(self.state.last_exit_dt)
        cd_remaining = self.risk.cooldown_remaining_hours(self.state.last_exit_dt)
        cycle_data.update({
            "in_cooldown":          in_cd,
            "cooldown_remaining_h": round(cd_remaining, 2),
        })
        if in_cd:
            log.info(f"In post-exit cooldown. {cd_remaining:.1f} h remaining.")

        # ── 5. Unrealised P&L (informational) ───────────────────────────
        if has_position and self.state.entry_price and self.state.entry_price > 0:
            upnl_pct = (btc_price - self.state.entry_price) / self.state.entry_price * 100
            cycle_data["entry_price"]       = self.state.entry_price
            cycle_data["unrealised_pnl_pct"] = round(upnl_pct, 3)
            log.info(
                f"Entry: ${self.state.entry_price:,.2f}  "
                f"uPnL: {upnl_pct:+.2f}%"
            )

        # ── 6. Determine action ──────────────────────────────────────────
        action = self._determine_action(signal, has_position, in_cd)
        cycle_data["action"] = action
        log.info(f"Action: {action}")

        # ── 7. Execute ───────────────────────────────────────────────────
        if action == "BUY":
            self._execute_buy(
                btc_price, usdt_balance, has_position, cycle_data
            )

        elif action == "SELL":
            self._execute_sell(
                btc_price, btc_balance, has_position, cycle_data
            )

        # ── 8. Write cycle log ───────────────────────────────────────────
        self.logger.log_cycle(cycle_data)
        return cycle_data

    # ------------------------------------------------------------------
    # Signal pipeline
    # ------------------------------------------------------------------

    def _compute_signal(self) -> tuple[str, str, int, float]:
        """
        Fetch Binance klines, run HMM + indicators, return:
          (signal, regime, votes, latest_close_price)
        """
        df = fetch_binance_klines(
            symbol=self.cfg.symbol,
            interval=self.cfg.interval,
            lookback_days=self.cfg.lookback_days,
            timeout=self.cfg.request_timeout,
        )

        # HMM features and training
        X, hmm_index = compute_hmm_features(df)
        hmm_cfg = HMMConfig(
            n_components=self.cfg.hmm_states,
            n_iter=200,
            random_state=42,
        )
        model = RegimeModel(hmm_cfg)
        model.fit(X)
        regimes = model.predict_series(X, hmm_index)

        # Indicators and signal
        ind_df = compute_indicators(df)

        # Use NORMAL_MODE with live min_votes override
        from config import ModeConfig
        live_mode = ModeConfig(
            name="Live",
            leverage=1.0,         # spot only — leverage is irrelevant
            min_votes=self.cfg.min_votes,
            trailing_stop_pct=None,
        )
        sig_df = generate_signals(ind_df, regimes, mode=live_mode)

        latest = sig_df.iloc[-1]
        signal = str(latest["signal"])
        regime = str(latest["regime"])
        votes  = int(latest["votes"])
        price  = float(latest["Close"])

        return signal, regime, votes, price

    # ------------------------------------------------------------------
    # Action logic
    # ------------------------------------------------------------------

    def _determine_action(
        self,
        signal:       str,
        has_position: bool,
        in_cooldown:  bool,
    ) -> str:
        """
        Pure decision function — no side effects.

        Rules
        -----
        LONG + no_position + not_cooldown → BUY
        CASH + has_position               → SELL
        everything else                   → HOLD
        """
        if signal == SIGNAL_LONG and not has_position and not in_cooldown:
            return "BUY"
        if signal == SIGNAL_CASH and has_position:
            return "SELL"
        return "HOLD"

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _execute_buy(
        self,
        btc_price:    float,
        usdt_balance: float,
        has_position: bool,
        cycle_data:   dict,
    ) -> None:
        """Place a market BUY order with full pre-flight checks."""
        log.info(
            f"Attempting BUY — notional=${self.cfg.max_notional_usd:.2f} USDT "
            f"at ~${btc_price:,.2f}"
        )

        # Pre-flight risk checks (raise on failure)
        try:
            self.risk.check_can_buy(
                usdt_balance=usdt_balance,
                has_position=has_position,
                last_trade_time=self.state.last_trade_dt,
                last_exit_time=self.state.last_exit_dt,
            )
        except ValueError as exc:
            log.warning(f"BUY pre-flight failed: {exc}")
            cycle_data.update({"action": "BUY_BLOCKED", "error": str(exc)})
            return

        # Place order
        try:
            order = self.broker.place_market_buy(price=btc_price)
        except Exception as exc:
            log.error(f"BUY order failed: {exc}")
            cycle_data.update({"action": "BUY_FAILED", "error": str(exc)[:300]})
            self.logger.log_trade({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "side":      "BUY",
                "symbol":    self.cfg.symbol,
                "status":    "FAILED",
                "error":     str(exc)[:300],
            })
            return

        executed_qty     = float(order.get("executedQty", 0))
        executed_notional = float(order.get("cummulativeQuoteQty", 0))
        avg_price         = (
            executed_notional / executed_qty if executed_qty > 0 else btc_price
        )

        log.info(
            f"BUY FILLED: {executed_qty:.6f} BTC "
            f"@ avg ${avg_price:,.2f} "
            f"(notional=${executed_notional:.2f}) "
            f"orderId={order.get('orderId')}"
        )

        # Update state
        self.state.record_entry(price=avg_price, units=executed_qty)

        cycle_data.update({
            "order_id":          order.get("orderId"),
            "executed_qty":      round(executed_qty, 8),
            "executed_notional": round(executed_notional, 4),
        })

        # Trade log
        self.logger.log_trade({
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "side":               "BUY",
            "symbol":             self.cfg.symbol,
            "order_id":           order.get("orderId"),
            "status":             order.get("status"),
            "executed_qty_btc":   executed_qty,
            "executed_notional":  executed_notional,
            "avg_price":          avg_price,
            "estimated_fee_usd":  self.risk.estimate_fees(executed_notional),
            "raw_order":          {k: order[k] for k in ("orderId", "status", "side", "type") if k in order},
        })

    def _execute_sell(
        self,
        btc_price:    float,
        btc_balance:  float,
        has_position: bool,
        cycle_data:   dict,
    ) -> None:
        """Place a market SELL (close full position) with full pre-flight checks."""
        log.info(
            f"Attempting SELL — close {btc_balance:.6f} BTC "
            f"at ~${btc_price:,.2f}"
        )

        # Pre-flight risk checks
        try:
            self.risk.check_can_sell(
                has_position=has_position,
                btc_units=btc_balance,
                btc_price=btc_price,
            )
        except ValueError as exc:
            log.warning(f"SELL pre-flight failed: {exc}")
            cycle_data.update({"action": "SELL_BLOCKED", "error": str(exc)})
            return

        # Place order
        try:
            result = self.broker.close_position(price=btc_price)
        except Exception as exc:
            log.error(f"SELL order failed: {exc}")
            cycle_data.update({"action": "SELL_FAILED", "error": str(exc)[:300]})
            self.logger.log_trade({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "side":      "SELL",
                "symbol":    self.cfg.symbol,
                "status":    "FAILED",
                "error":     str(exc)[:300],
            })
            return

        if result.get("status") == "no_position":
            log.warning("SELL: no position found on exchange (already flat?)")
            cycle_data.update({"action": "SELL_NOOP"})
            return

        executed_qty      = float(result.get("units_sold", 0))
        proceeds          = float(result.get("proceeds_usdt", 0))
        avg_price         = proceeds / executed_qty if executed_qty > 0 else btc_price

        # P&L vs tracked entry
        pnl_usd = 0.0
        pnl_pct = 0.0
        if self.state.entry_price and self.state.entry_units:
            cost    = self.state.entry_price * self.state.entry_units
            pnl_usd = proceeds - cost
            pnl_pct = pnl_usd / cost * 100 if cost > 0 else 0.0

        log.info(
            f"SELL FILLED: {executed_qty:.6f} BTC "
            f"@ avg ${avg_price:,.2f} "
            f"proceeds=${proceeds:.2f}  "
            f"PnL={pnl_pct:+.2f}%  "
            f"orderId={result.get('order_id')}"
        )

        # Update state
        self.state.record_exit()

        cycle_data.update({
            "order_id":          result.get("order_id"),
            "executed_qty":      round(executed_qty, 8),
            "executed_notional": round(proceeds, 4),
        })

        # Trade log
        self.logger.log_trade({
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "side":              "SELL",
            "symbol":            self.cfg.symbol,
            "order_id":          result.get("order_id"),
            "status":            "FILLED",
            "executed_qty_btc":  executed_qty,
            "proceeds_usdt":     proceeds,
            "avg_price":         avg_price,
            "entry_price":       self.state.entry_price,
            "pnl_usd":           round(pnl_usd, 4),
            "pnl_pct":           round(pnl_pct, 4),
            "estimated_fee_usd": self.risk.estimate_fees(proceeds),
            "raw_order":         result.get("raw_order", {}),
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Entry point. Validates environment then starts the live trading loop.
    """
    load_dotenv()

    print("=" * 60)
    print("  BTC Regime Trader — Live Mode")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    # Check API keys before anything else
    api_key    = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")

    if not api_key or api_key == "your_binance_api_key_here":
        print("\n[ERROR] BINANCE_API_KEY is not set or is the placeholder value.")
        print("  1. Copy .env.example to .env")
        print("  2. Add your real Binance API key")
        print("  3. Re-run this script")
        sys.exit(1)

    if not api_secret or api_secret == "your_binance_api_secret_here":
        print("\n[ERROR] BINANCE_API_SECRET is not set or is the placeholder value.")
        sys.exit(1)

    live_enabled = os.environ.get("LIVE_TRADING_ENABLED", "false").lower()
    live_confirm = os.environ.get("LIVE_CONFIRM", "")

    print(f"\n  API key      : {api_key[:8]}...{api_key[-4:]} (masked)")
    print(f"  Live enabled : {live_enabled}")
    print(f"  Live confirm : {'SET ✓' if live_confirm == 'YES_LIVE_TRADING' else 'NOT SET ✗'}")

    if live_enabled != "true":
        print("\n[WARNING] LIVE_TRADING_ENABLED is not 'true'.")
        print("  The bot will run in monitor-only mode (no orders will be placed).")
        print("  To enable live trading, set LIVE_TRADING_ENABLED=true in your .env")

    if live_confirm != "YES_LIVE_TRADING" and live_enabled == "true":
        print("\n[WARNING] LIVE_CONFIRM is not set to 'YES_LIVE_TRADING'.")
        print("  No orders will be placed until this is set.")

    print()

    trader = LiveTrader()
    trader.run()


if __name__ == "__main__":
    main()
