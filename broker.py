"""
broker.py
=========
Broker interface abstraction.

Classes
-------
BaseBroker          – abstract interface every broker must implement
PaperBroker         – in-memory simulation with optional JSON state persistence
BinanceSpotBroker   – real Binance Spot REST API broker (BTCUSDT, no futures/margin)

Safety notes for BinanceSpotBroker
-----------------------------------
• Uses raw REST API + HMAC-SHA256 signing (no third-party client library)
• Market BUY  uses quoteOrderQty  (spend exactly N USDT → receive BTC)
• Market SELL uses quantity        (sell exactly N BTC → receive USDT)
• All quantities are floored to stepSize before submission
• min_notional is validated before every order
• A hard max_notional_usd cap is applied to every BUY order
• recvWindow and timestamp are attached to every signed request
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
import urllib.parse
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests as _requests

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseBroker(ABC):
    """Every broker (paper or live) must implement these core methods."""

    @abstractmethod
    def get_balance(self) -> float:
        """Return available cash balance in USD / USDT."""

    @abstractmethod
    def get_position(self) -> Optional[dict]:
        """Return the current open position dict, or None if flat."""

    @abstractmethod
    def place_market_buy(self, price: float, units: float) -> dict:
        """Open a long position. Returns an order confirmation dict."""

    @abstractmethod
    def place_market_sell(self, price: float, units: float) -> dict:
        """Partially close a long position. Returns order confirmation dict."""

    @abstractmethod
    def close_position(self, price: float) -> dict:
        """Close the entire open position at `price`. Returns trade summary dict."""


# ---------------------------------------------------------------------------
# Paper broker
# ---------------------------------------------------------------------------

class PaperBroker(BaseBroker):
    """
    In-memory paper broker with optional JSON state persistence.

    Parameters
    ----------
    starting_capital : float – initial USD balance
    persist          : bool  – if True, saves/loads state from STATE_FILE
    state_file       : str   – override the default state file path
    """

    DEFAULT_STATE_FILE = Path("paper_broker_state.json")

    def __init__(
        self,
        starting_capital: float = 10_000.0,
        persist:          bool  = True,
        state_file:       Optional[str] = None,
    ):
        self._starting_capital = starting_capital
        self._balance: float         = starting_capital
        self._position: Optional[dict] = None
        self._persist  = persist
        self._state_file = Path(state_file) if state_file else self.DEFAULT_STATE_FILE

        if persist and self._state_file.exists():
            self._load_state()

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        return round(self._balance, 2)

    def get_position(self) -> Optional[dict]:
        return self._position

    def get_equity(self, current_price: float) -> float:
        """Total equity = cash + position mark-to-market value."""
        if self._position is None:
            return self._balance
        return self._balance + self._position["units"] * current_price

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def place_market_buy(self, price: float, units: float) -> dict:
        cost = price * units
        if cost > self._balance + 1e-6:
            raise ValueError(
                f"Insufficient balance (${self._balance:.2f}) "
                f"for buy order (${cost:.2f})"
            )
        self._balance  -= cost
        self._position  = {
            "entry_price": price,
            "units":       units,
            "entry_time":  datetime.now(timezone.utc).isoformat(),
            "cost":        cost,
        }
        self._save_state()
        return {"side": "BUY", "price": price, "units": units, "cost": cost}

    def place_market_sell(self, price: float, units: float) -> dict:
        if self._position is None:
            raise ValueError("No open position to sell.")
        units     = min(units, self._position["units"])
        proceeds  = price * units
        self._balance += proceeds
        remaining     = self._position["units"] - units
        if remaining <= 1e-9:
            self._position = None
        else:
            self._position["units"] = remaining
        self._save_state()
        return {"side": "SELL", "price": price, "units": units, "proceeds": proceeds}

    def close_position(self, price: float) -> dict:
        if self._position is None:
            return {"status": "no_position"}

        pos      = self._position
        units    = pos["units"]
        proceeds = price * units
        pnl      = proceeds - pos["cost"]

        self._balance += proceeds
        self._position = None
        self._save_state()

        return {
            "status":      "closed",
            "entry_price": pos["entry_price"],
            "exit_price":  price,
            "units":       units,
            "proceeds":    proceeds,
            "pnl":         pnl,
            "entry_time":  pos["entry_time"],
            "exit_time":   datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self, starting_capital: Optional[float] = None):
        """Reset to a clean state (wipes position and resets balance)."""
        self._balance  = starting_capital or self._starting_capital
        self._position = None
        self._save_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self):
        if not self._persist:
            return
        state = {"balance": self._balance, "position": self._position}
        try:
            self._state_file.write_text(json.dumps(state, indent=2))
        except OSError:
            pass

    def _load_state(self):
        try:
            state          = json.loads(self._state_file.read_text())
            self._balance  = float(state.get("balance", self._balance))
            self._position = state.get("position")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Binance Spot broker — real live trading
# ---------------------------------------------------------------------------

class BinanceSpotBroker(BaseBroker):
    """
    Real Binance Spot REST API broker.

    Design principles
    -----------------
    • Spot only — no futures, no margin, no leverage
    • Market BUY  via quoteOrderQty  (spend X USDT, receive BTC)
    • Market SELL via quantity        (sell N BTC, receive USDT)
    • All quantities floored to stepSize (never rounded up)
    • Hard cap applied to every BUY: notional <= max_notional_usd
    • min_notional validated before every order
    • Symbol info cached per instance; call refresh_symbol_info() to reload
    • Every error is raised with a descriptive message — never silently swallowed

    Parameters
    ----------
    api_key         : Binance API key (read from env — do NOT hardcode)
    api_secret      : Binance API secret (read from env — do NOT hardcode)
    symbol          : Default "BTCUSDT"
    max_notional_usd: Hard cap on BUY order size in USD (default 50)
    recv_window     : Binance recvWindow in milliseconds (default 5000)
    timeout         : HTTP request timeout in seconds (default 10)
    dust_btc        : BTC balance below this is treated as flat (default 0.00005)
    testnet         : If True, uses https://testnet.binance.vision instead
    """

    _LIVE_URL    = "https://api.binance.com"
    _TESTNET_URL = "https://testnet.binance.vision"

    def __init__(
        self,
        api_key:          str,
        api_secret:       str,
        symbol:           str   = "BTCUSDT",
        max_notional_usd: float = 50.0,
        recv_window:      int   = 5000,
        timeout:          int   = 10,
        dust_btc:         float = 0.00005,
        testnet:          bool  = False,
    ):
        if not api_key or not api_secret:
            raise ValueError(
                "BINANCE_API_KEY and BINANCE_API_SECRET must be set. "
                "Load them from environment variables — never hardcode secrets."
            )

        self._api_key          = api_key
        self._api_secret       = api_secret
        self._symbol           = symbol.upper()
        self._base_asset       = self._symbol.replace("USDT", "")  # e.g. "BTC"
        self._max_notional_usd = max_notional_usd
        self._recv_window      = recv_window
        self._timeout          = timeout
        self._dust_btc         = dust_btc
        self._base_url         = self._TESTNET_URL if testnet else self._LIVE_URL
        self._symbol_info_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public read methods
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        """Return free USDT balance."""
        account = self._signed_get("/api/v3/account")
        for bal in account.get("balances", []):
            if bal["asset"] == "USDT":
                return float(bal["free"])
        return 0.0

    def get_position(self) -> Optional[dict]:
        """
        Return a position dict if we hold a meaningful BTC balance, else None.

        Note: Binance Spot has no "position" concept. We check the BTC balance.
        Entry price / time are tracked externally by live_trader.py.

        Returns
        -------
        {"units": float, "asset": "BTC"} or None
        """
        account = self._signed_get("/api/v3/account")
        btc_free = 0.0
        for bal in account.get("balances", []):
            if bal["asset"] == self._base_asset:
                btc_free = float(bal["free"])
                break

        if btc_free >= self._dust_btc:
            return {"units": btc_free, "asset": self._base_asset}
        return None

    def get_account_snapshot(self) -> dict:
        """Return USDT balance + BTC balance in one signed call."""
        account = self._signed_get("/api/v3/account")
        usdt_free = 0.0
        btc_free  = 0.0
        for bal in account.get("balances", []):
            if bal["asset"] == "USDT":
                usdt_free = float(bal["free"])
            elif bal["asset"] == self._base_asset:
                btc_free = float(bal["free"])
        return {
            "usdt_free": usdt_free,
            "btc_free":  btc_free,
            "has_position": btc_free >= self._dust_btc,
        }

    def get_ticker_price(self) -> float:
        """Return the latest price for the symbol (public endpoint, no auth)."""
        resp = _requests.get(
            f"{self._base_url}/api/v3/ticker/price",
            params={"symbol": self._symbol},
            timeout=self._timeout,
        )
        self._check_http(resp)
        return float(resp.json()["price"])

    # ------------------------------------------------------------------
    # Symbol info and filters
    # ------------------------------------------------------------------

    def get_symbol_info(self, force_refresh: bool = False) -> dict:
        """
        Return the exchange info dict for the symbol (cached per instance).
        Contains all trading filters: LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL, etc.
        """
        if self._symbol_info_cache is None or force_refresh:
            self._symbol_info_cache = self._fetch_symbol_info()
        return self._symbol_info_cache

    def refresh_symbol_info(self) -> dict:
        """Force-refresh the symbol info cache and return it."""
        return self.get_symbol_info(force_refresh=True)

    def _fetch_symbol_info(self) -> dict:
        resp = _requests.get(
            f"{self._base_url}/api/v3/exchangeInfo",
            params={"symbol": self._symbol},
            timeout=self._timeout,
        )
        self._check_http(resp)
        data = resp.json()
        for sym in data.get("symbols", []):
            if sym["symbol"] == self._symbol:
                return sym
        raise ValueError(
            f"Symbol '{self._symbol}' not found in Binance exchange info. "
            "Check that the symbol is valid and trading is enabled."
        )

    def _get_filter(self, filter_type: str) -> Optional[dict]:
        """Extract a named filter from the cached symbol info."""
        info = self.get_symbol_info()
        for f in info.get("filters", []):
            if f["filterType"] == filter_type:
                return f
        return None

    def get_lot_size_filter(self) -> dict:
        """Return LOT_SIZE filter dict (minQty, maxQty, stepSize)."""
        f = self._get_filter("LOT_SIZE")
        if f is None:
            raise RuntimeError(f"No LOT_SIZE filter found for {self._symbol}")
        return f

    def get_min_notional_filter(self) -> Optional[dict]:
        """Return MIN_NOTIONAL or NOTIONAL filter dict, or None if not present."""
        for filter_type in ("MIN_NOTIONAL", "NOTIONAL"):
            f = self._get_filter(filter_type)
            if f is not None:
                return f
        return None

    # ------------------------------------------------------------------
    # Quantity and price rounding
    # ------------------------------------------------------------------

    def round_quantity_to_step_size(self, quantity: float, step_size: str) -> float:
        """
        Floor quantity to the nearest valid step.

        Always floors (never rounds up) to avoid 'insufficient balance' errors.
        Uses string-based precision to avoid float rounding artifacts.
        """
        step = float(step_size)
        if step <= 0:
            return quantity

        # Compute decimal precision from stepSize string (e.g. "0.00001000" → 5)
        step_str = step_size.rstrip("0")
        if "." in step_str:
            precision = len(step_str.split(".")[-1])
        else:
            precision = 0

        floored = math.floor(quantity / step) * step
        return round(floored, precision)

    def round_price_to_tick_size(self, price: float, tick_size: str) -> float:
        """Round price to the nearest valid tick (standard rounding)."""
        tick = float(tick_size)
        if tick <= 0:
            return price

        tick_str = tick_size.rstrip("0")
        if "." in tick_str:
            precision = len(tick_str.split(".")[-1])
        else:
            precision = 0

        rounded = round(price / tick) * tick
        return round(rounded, precision)

    def validate_min_notional(self, quantity: float, price: float) -> bool:
        """Return True if quantity * price meets the MIN_NOTIONAL requirement."""
        f = self.get_min_notional_filter()
        if f is None:
            return True  # No filter → always valid
        min_notional = float(f.get("minNotional", f.get("minNotional", 10.0)))
        notional = quantity * price
        if notional < min_notional:
            raise ValueError(
                f"Order notional ${notional:.4f} is below MIN_NOTIONAL ${min_notional:.4f}. "
                f"Increase order size or check dust threshold."
            )
        return True

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def place_market_buy(self, price: float, units: float = 0.0) -> dict:
        """
        Place a MARKET BUY using quoteOrderQty (spend USDT, receive BTC).

        The `units` parameter is ignored for live orders — we always spend
        self._max_notional_usd USDT. Providing `price` is optional for the
        live broker (kept for interface compatibility with PaperBroker).

        Returns
        -------
        Binance order response dict with keys:
          orderId, status, executedQty, cummulativeQuoteQty, fills, ...
        """
        notional = self._max_notional_usd

        # Safety: cap at max_notional_usd (redundant but defensive)
        notional = min(notional, self._max_notional_usd)
        if notional <= 0:
            raise ValueError(f"BUY notional must be > 0, got {notional}")

        # Validate min notional against exchange filters
        # For quoteOrderQty orders, min_notional = quoteOrderQty
        f = self.get_min_notional_filter()
        if f is not None:
            min_notional = float(f.get("minNotional", 10.0))
            if notional < min_notional:
                raise ValueError(
                    f"BUY notional ${notional:.2f} is below exchange MIN_NOTIONAL "
                    f"${min_notional:.2f}. Increase MAX_LIVE_NOTIONAL_USD."
                )

        params = {
            "symbol":        self._symbol,
            "side":          "BUY",
            "type":          "MARKET",
            "quoteOrderQty": f"{notional:.2f}",
        }

        order = self._signed_post("/api/v3/order", params)
        self._assert_order_filled(order)
        return order

    def place_market_sell(self, price: float, units: float) -> dict:
        """
        Place a MARKET SELL for exactly `units` BTC.

        Quantity is floored to stepSize before submission.
        min_notional is validated using the provided `price` estimate.

        Returns
        -------
        Binance order response dict.
        """
        if units <= 0:
            raise ValueError(f"SELL units must be > 0, got {units}")

        lot = self.get_lot_size_filter()
        step_size  = lot["stepSize"]
        min_qty    = float(lot["minQty"])
        max_qty    = float(lot["maxQty"])

        qty = self.round_quantity_to_step_size(units, step_size)

        if qty <= 0:
            raise ValueError(
                f"After rounding to stepSize={step_size}, quantity is 0. "
                f"Input units={units:.8f} is below minimum lot size."
            )
        if qty < min_qty:
            raise ValueError(
                f"Quantity {qty:.8f} BTC is below LOT_SIZE minQty {min_qty:.8f}."
            )
        if qty > max_qty:
            raise ValueError(
                f"Quantity {qty:.8f} BTC exceeds LOT_SIZE maxQty {max_qty:.8f}."
            )

        # Validate min notional
        self.validate_min_notional(qty, price)

        # Format quantity with correct precision
        step_str = step_size.rstrip("0")
        precision = len(step_str.split(".")[-1]) if "." in step_str else 0
        qty_str = f"{qty:.{precision}f}"

        params = {
            "symbol":   self._symbol,
            "side":     "SELL",
            "type":     "MARKET",
            "quantity": qty_str,
        }

        order = self._signed_post("/api/v3/order", params)
        self._assert_order_filled(order)
        return order

    def close_position(self, price: float) -> dict:
        """
        Sell the entire BTC balance. If no meaningful position, returns safely.

        Returns
        -------
        dict with status, executedQty, proceeds, and order details.
        """
        pos = self.get_position()
        if pos is None:
            return {"status": "no_position", "message": "No BTC position to close."}

        units = pos["units"]
        order = self.place_market_sell(price=price, units=units)

        executed_qty   = float(order.get("executedQty", 0))
        quote_qty      = float(order.get("cummulativeQuoteQty", executed_qty * price))

        return {
            "status":        "closed",
            "order_id":      order.get("orderId"),
            "symbol":        self._symbol,
            "units_sold":    executed_qty,
            "proceeds_usdt": quote_qty,
            "exit_time":     datetime.now(timezone.utc).isoformat(),
            "raw_order":     order,
        }

    # ------------------------------------------------------------------
    # Signed REST helpers
    # ------------------------------------------------------------------

    def _sign_params(self, params: dict) -> str:
        """
        Attach timestamp + recvWindow, compute HMAC-SHA256 signature,
        and return the full signed query string.
        """
        p = dict(params)  # copy — do not mutate caller's dict
        p["timestamp"]  = int(time.time() * 1000)
        p["recvWindow"] = self._recv_window

        qs = urllib.parse.urlencode(p)
        sig = hmac.new(
            self._api_secret.encode("utf-8"),
            qs.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return f"{qs}&signature={sig}"

    def _signed_get(self, path: str, params: Optional[dict] = None) -> dict:
        signed_qs = self._sign_params(params or {})
        url = f"{self._base_url}{path}?{signed_qs}"
        resp = _requests.get(
            url,
            headers={"X-MBX-APIKEY": self._api_key},
            timeout=self._timeout,
        )
        self._check_http(resp)
        return resp.json()

    def _signed_post(self, path: str, params: dict) -> dict:
        signed_qs = self._sign_params(params)
        url = f"{self._base_url}{path}?{signed_qs}"
        resp = _requests.post(
            url,
            headers={"X-MBX-APIKEY": self._api_key},
            timeout=self._timeout,
        )
        self._check_http(resp)
        return resp.json()

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _check_http(self, resp: _requests.Response) -> None:
        """
        Raise a descriptive exception for any non-2xx response,
        parsing Binance's error codes where possible.
        """
        if resp.ok:
            return

        # Try to parse Binance error body
        try:
            body = resp.json()
            code = body.get("code", resp.status_code)
            msg  = body.get("msg",  resp.text)
        except Exception:
            code = resp.status_code
            msg  = resp.text

        error_map = {
            -1000: "UNKNOWN — An unknown error occurred.",
            -1003: "TOO_MANY_REQUESTS — Rate limit exceeded. Back off.",
            -1013: "INVALID_QTY — Filter failure (lot size / min notional).",
            -1100: "ILLEGAL_CHARACTERS — Invalid parameter value.",
            -1111: "BAD_PRECISION — Quantity/price precision issue.",
            -1121: "INVALID_SYMBOL — Symbol not found on exchange.",
            -2010: "NEW_ORDER_REJECTED — Insufficient balance or filter failure.",
            -2011: "CANCEL_REJECTED — Order not found or already cancelled.",
            -1022: "INVALID_SIGNATURE — Signature mismatch. Check API secret.",
            -1102: "MANDATORY_PARAM_EMPTY — A required parameter is missing.",
        }

        hint = error_map.get(int(code) if str(code).lstrip("-").isdigit() else 0, "")
        raise RuntimeError(
            f"Binance API error {code}: {msg}"
            + (f"\nHint: {hint}" if hint else "")
        )

    def _assert_order_filled(self, order: dict) -> None:
        """Raise if the order was not filled (MARKET orders should always fill)."""
        status = order.get("status", "UNKNOWN")
        if status not in ("FILLED", "PARTIALLY_FILLED"):
            raise RuntimeError(
                f"Order {order.get('orderId')} did not fill. "
                f"Status: {status}. Full response: {order}"
            )
