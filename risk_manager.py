"""
risk_manager.py
===============
Two risk managers:

  RiskManager      – original stateless helper for backtesting / paper trading
  LiveRiskManager  – live-specific checks for real Binance spot execution

The live manager is intentionally conservative:
  • Hard cap on notional size
  • One position at a time enforcement
  • Cooldown after exits
  • Minimum time between any two trades
  • Fee + slippage estimates for logging purposes
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config import ModeConfig, RiskConfig, LiveConfig, NORMAL_MODE, DEFAULT_CONFIG
from regime_model import BEAR


# ===========================================================================
# Original risk manager  (backtest / paper trading)
# ===========================================================================

class RiskManager:
    """
    Parameters
    ----------
    risk_cfg : RiskConfig  — fee, slippage, cooldown, capital fraction
    mode_cfg : ModeConfig  — leverage, trailing stop pct
    """

    def __init__(
        self,
        risk_cfg: RiskConfig  = DEFAULT_CONFIG.risk,
        mode_cfg: ModeConfig  = NORMAL_MODE,
    ):
        self.risk = risk_cfg
        self.mode = mode_cfg

    # ------------------------------------------------------------------
    # Exit decision
    # ------------------------------------------------------------------

    def should_exit(
        self,
        current_regime: str,
        current_price:  float,
        peak_price:     float,
    ) -> tuple[bool, str]:
        """
        Priority
        --------
        1. Bear-regime hard exit
        2. Trailing stop
        """
        if current_regime == BEAR:
            return True, "Bear Regime"

        if self.mode.trailing_stop_pct is not None and peak_price > 0:
            stop_level = peak_price * (1.0 - self.mode.trailing_stop_pct / 100.0)
            if current_price < stop_level:
                return True, f"Trailing Stop ({self.mode.trailing_stop_pct:.0f}%)"

        return False, ""

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------

    def in_cooldown(
        self,
        last_exit_time: Optional[pd.Timestamp],
        current_time:   pd.Timestamp,
    ) -> bool:
        """Return True if still within the post-exit cooldown window."""
        if last_exit_time is None:
            return False
        elapsed_hours = (current_time - last_exit_time).total_seconds() / 3_600
        return elapsed_hours < self.risk.cooldown_hours

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def position_size(self, capital: float) -> float:
        """Capital (USD) to allocate to the next trade, before leverage."""
        return capital * self.risk.max_position_fraction

    # ------------------------------------------------------------------
    # Trade-cost simulation
    # ------------------------------------------------------------------

    def adjusted_entry_price(self, raw_price: float) -> float:
        """Apply slippage and fee on entry."""
        slipped = raw_price * (1.0 + self.risk.slippage_pct / 100.0)
        fee     = slipped   *  self.risk.fee_pct           / 100.0
        return slipped + fee

    def adjusted_exit_price(self, raw_price: float) -> float:
        """Apply slippage and fee on exit."""
        slipped = raw_price * (1.0 - self.risk.slippage_pct / 100.0)
        fee     = slipped   *  self.risk.fee_pct             / 100.0
        return slipped - fee

    # ------------------------------------------------------------------
    # P&L helpers
    # ------------------------------------------------------------------

    def compute_leveraged_return(self, entry_price: float, exit_price: float) -> float:
        if entry_price <= 0:
            return 0.0
        raw_ret = (exit_price - entry_price) / entry_price
        lev_ret = raw_ret * self.mode.leverage
        return max(lev_ret, -1.0)

    def apply_pnl(
        self, entry_capital: float, entry_price: float, exit_price: float
    ) -> float:
        lev_ret = self.compute_leveraged_return(entry_price, exit_price)
        return max(entry_capital * (1.0 + lev_ret), 0.0)


# ===========================================================================
# Live risk manager  (Binance Spot execution)
# ===========================================================================

class LiveRiskManager:
    """
    All risk checks that must pass before a real Binance order is placed.

    Safety design
    -------------
    • check_can_buy()  and  check_can_sell()  raise ValueError with a clear
      reason if ANY check fails.
    • The caller (live_trader.py) wraps these in try/except and logs the reason.
    • No check is silently skipped.

    Parameters
    ----------
    cfg : LiveConfig — live trading parameters from config.py
    """

    def __init__(self, cfg: LiveConfig = DEFAULT_CONFIG.live):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Kill switch validation (always call this first)
    # ------------------------------------------------------------------

    def assert_live_enabled(self, live_confirm_env: str) -> None:
        """
        Assert that BOTH the config kill switch AND the environment
        confirmation variable are set correctly.

        Parameters
        ----------
        live_confirm_env : value of the LIVE_CONFIRM environment variable

        Raises
        ------
        RuntimeError if live trading is not fully enabled.
        """
        if not self.cfg.live_trading_enabled:
            raise RuntimeError(
                "LIVE TRADING DISABLED: config.live.live_trading_enabled = False.\n"
                "Set LIVE_TRADING_ENABLED=true in your .env file to enable."
            )

        if self.cfg.require_explicit_confirmation:
            if live_confirm_env != "YES_LIVE_TRADING":
                raise RuntimeError(
                    "LIVE TRADING DISABLED: LIVE_CONFIRM env var must equal "
                    "'YES_LIVE_TRADING' exactly.\n"
                    "Add LIVE_CONFIRM=YES_LIVE_TRADING to your .env file."
                )

    # ------------------------------------------------------------------
    # BUY pre-flight checks
    # ------------------------------------------------------------------

    def check_can_buy(
        self,
        usdt_balance:   float,
        has_position:   bool,
        last_trade_time: Optional[datetime],
        last_exit_time:  Optional[datetime],
    ) -> None:
        """
        Validate all conditions for a BUY order.

        Raises ValueError with a human-readable reason if any check fails.
        """
        if has_position:
            raise ValueError(
                "BUY BLOCKED: already holding a BTC position. "
                "No pyramiding allowed."
            )

        notional = self.cfg.max_notional_usd
        if usdt_balance < notional:
            raise ValueError(
                f"BUY BLOCKED: insufficient USDT balance "
                f"(have ${usdt_balance:.2f}, need ${notional:.2f})."
            )

        if usdt_balance < 11.0:
            raise ValueError(
                f"BUY BLOCKED: USDT balance ${usdt_balance:.2f} too low for minimum order."
            )

        # Minimum time between trades
        if last_trade_time is not None:
            elapsed_min = (
                datetime.now(timezone.utc) - last_trade_time
            ).total_seconds() / 60
            if elapsed_min < self.cfg.min_minutes_between_trades:
                remaining = self.cfg.min_minutes_between_trades - elapsed_min
                raise ValueError(
                    f"BUY BLOCKED: minimum trade interval not elapsed. "
                    f"{remaining:.0f} min remaining (min={self.cfg.min_minutes_between_trades} min)."
                )

        # Post-exit cooldown
        if last_exit_time is not None:
            elapsed_h = (
                datetime.now(timezone.utc) - last_exit_time
            ).total_seconds() / 3600
            if elapsed_h < self.cfg.cooldown_hours:
                remaining_h = self.cfg.cooldown_hours - elapsed_h
                raise ValueError(
                    f"BUY BLOCKED: post-exit cooldown active. "
                    f"{remaining_h:.1f} h remaining (cooldown={self.cfg.cooldown_hours} h)."
                )

    # ------------------------------------------------------------------
    # SELL pre-flight checks
    # ------------------------------------------------------------------

    def check_can_sell(
        self,
        has_position: bool,
        btc_units:    float,
        btc_price:    float,
    ) -> None:
        """
        Validate all conditions for a SELL (close) order.

        Raises ValueError if any check fails.
        """
        if not has_position:
            raise ValueError("SELL BLOCKED: no BTC position to close.")

        if btc_units <= 0:
            raise ValueError(
                f"SELL BLOCKED: BTC units is {btc_units:.8f} — nothing to sell."
            )

        notional = btc_units * btc_price
        if notional < 10.0:
            raise ValueError(
                f"SELL BLOCKED: notional ${notional:.2f} is below $10 minimum. "
                f"May fall below exchange MIN_NOTIONAL filter."
            )

    # ------------------------------------------------------------------
    # Notional cap (apply on every BUY)
    # ------------------------------------------------------------------

    def apply_notional_cap(self, requested_notional: float) -> float:
        """Return the effective notional, capped at max_notional_usd."""
        capped = min(requested_notional, self.cfg.max_notional_usd)
        if capped < requested_notional:
            pass  # caller can log this
        return capped

    # ------------------------------------------------------------------
    # Cost estimates (for cycle logging — real fees charged by exchange)
    # ------------------------------------------------------------------

    def estimate_fees(self, notional: float) -> float:
        """Estimated exchange fee for a one-way trade (USD)."""
        return notional * self.cfg.fee_pct / 100.0

    def estimate_slippage(self, notional: float) -> float:
        """Estimated slippage cost for a one-way trade (USD)."""
        return notional * self.cfg.slippage_pct / 100.0

    def estimate_round_trip_cost(self, notional: float) -> float:
        """Estimated total cost (fees + slippage) for entry + exit (USD)."""
        return 2.0 * (self.estimate_fees(notional) + self.estimate_slippage(notional))

    # ------------------------------------------------------------------
    # Cooldown status  (informational, does not raise)
    # ------------------------------------------------------------------

    def cooldown_remaining_hours(
        self, last_exit_time: Optional[datetime]
    ) -> float:
        """Return hours remaining in cooldown, or 0 if not in cooldown."""
        if last_exit_time is None:
            return 0.0
        elapsed_h = (
            datetime.now(timezone.utc) - last_exit_time
        ).total_seconds() / 3600
        return max(0.0, self.cfg.cooldown_hours - elapsed_h)

    def is_in_cooldown(self, last_exit_time: Optional[datetime]) -> bool:
        return self.cooldown_remaining_hours(last_exit_time) > 0
