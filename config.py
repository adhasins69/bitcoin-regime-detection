"""
config.py
=========
Single source of truth for every tunable parameter in the pipeline.

All other modules import from this file — never hard-code numbers elsewhere.
Modify values here to tune the system without touching any logic code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Controls what data is downloaded from yfinance."""
    ticker:   str = "BTC-USD"
    period:   str = "730d"     # yfinance period string (2 years of hourly data)
    interval: str = "1h"       # candle granularity


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

@dataclass
class HMMConfig:
    """Gaussian HMM hyperparameters."""
    n_components:    int = 5          # number of hidden states (default 5)
    covariance_type: str = "full"     # full, diag, tied, or spherical
    n_iter:          int = 200        # maximum EM iterations
    random_state:    int = 42         # reproducibility seed


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """
    Controls signal generation thresholds.

    mode      : "Normal" (7/8 votes) or "Aggressive" (5/8 votes)
    min_votes : required confirmation count out of 8 conditions
    """
    mode:      str = "Normal"
    min_votes: int = 7

    # Individual confirmation thresholds (kept here for easy adjustment)
    rsi_max:        float = 90.0   # condition 1: RSI < rsi_max
    momentum_min:   float = 1.0    # condition 2: momentum_pct > momentum_min
    vol_max:        float = 6.0    # condition 3: volatility_pct < vol_max
    adx_min:        float = 25.0   # condition 5: ADX > adx_min


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Controls portfolio simulation assumptions."""
    starting_capital: float = 10_000.0
    fee_pct:          float = 0.10   # % per trade side (0.10 = 10 bps)
    slippage_pct:     float = 0.05   # % per trade side (0.05 = 5 bps)


# ---------------------------------------------------------------------------
# Root application config
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """
    Aggregated configuration object threaded through the full pipeline.

    Instantiate once and pass to BacktestEngine, or use DEFAULT_CONFIG.
    """
    data:     DataConfig     = field(default_factory=DataConfig)
    hmm:      HMMConfig      = field(default_factory=HMMConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


# Pre-built mode presets — import and use directly in the Streamlit sidebar.
NORMAL_MODE = StrategyConfig(mode="Normal", min_votes=7)
AGGRESSIVE_MODE = StrategyConfig(mode="Aggressive", min_votes=5)

# Global default — imported by modules that need sensible fallback values.
DEFAULT_CONFIG = AppConfig()
