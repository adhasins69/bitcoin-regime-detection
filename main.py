"""
main.py
=======
Command-line entry point for the Bitcoin Regime Detection and Backtesting System.

Runs the full pipeline headlessly (no Streamlit required):
  1. Download BTC-USD hourly data
  2. Engineer statistical and technical features
  3. Train the Gaussian HMM regime model
  4. Generate strategy signals
  5. Backtest the strategy
  6. Print performance metrics
  7. Save equity curve and trade log to outputs/

Usage
-----
    python main.py                         # Normal mode, default settings
    python main.py --mode aggressive       # Aggressive mode (5/8 votes)
    python main.py --states 7              # 7 HMM hidden states
    python main.py --capital 50000         # Custom starting capital
    python main.py --fee 0.05 --slip 0.02  # Custom costs
    python main.py --help                  # Show all options
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Make sure the project root is on the path (handles running from any directory)
sys.path.insert(0, str(Path(__file__).parent))

from config import AppConfig, DataConfig, HMMConfig, StrategyConfig, BacktestConfig
from data_loader import fetch_data
from backtester import BacktestEngine, compute_metrics, trades_to_dataframe

OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bitcoin Market Regime Detection and Backtesting System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["normal", "aggressive"], default="normal",
        help="Strategy mode. Normal = 7/8 votes. Aggressive = 5/8 votes.",
    )
    p.add_argument(
        "--states", type=int, default=5,
        help="Number of HMM hidden states.",
    )
    p.add_argument(
        "--capital", type=float, default=10_000.0,
        help="Starting capital in USD.",
    )
    p.add_argument(
        "--fee", type=float, default=0.10,
        help="Exchange fee per trade side, in percent (e.g. 0.10 = 10 bps).",
    )
    p.add_argument(
        "--slip", type=float, default=0.05,
        help="Estimated slippage per trade side, in percent.",
    )
    p.add_argument(
        "--period", type=str, default="730d",
        help="yfinance period string for data download (e.g. '730d', '365d').",
    )
    return p


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "─", width: int = 60) -> str:
    return char * width


def _print_metrics(metrics: dict) -> None:
    print()
    print(_sep("═"))
    print("  BACKTEST RESULTS — Bitcoin Regime Detection System")
    print(_sep("═"))
    print()

    ret   = metrics["total_return_pct"]
    bh    = metrics["bh_return_pct"]
    alpha = metrics["alpha"]
    dd    = metrics["max_drawdown"]
    wr    = metrics["win_rate"]
    sh    = metrics["sharpe"]
    n     = metrics["num_trades"]
    fv    = metrics["final_value"]
    sc    = metrics["start_capital"]
    mode  = metrics["mode"]
    mv    = metrics["min_votes"]
    sig   = metrics["current_signal"]
    reg   = metrics["current_regime"]
    votes = metrics["current_votes"]

    sign = lambda v: "+" if v >= 0 else ""

    print(f"  Mode           : {mode} (min {mv}/8 votes)")
    print(f"  Capital        : ${sc:,.2f} → ${fv:,.2f}")
    print()
    print(_sep())
    print(f"  Total Return   : {sign(ret)}{ret:.2f}%")
    print(f"  Buy & Hold     : {sign(bh)}{bh:.2f}%")
    print(f"  Alpha vs B&H   : {sign(alpha)}{alpha:.2f}%")
    print(_sep())
    print(f"  Win Rate       : {wr:.1f}%  ({n} trades)")
    print(f"  Max Drawdown   : {dd:.2f}%")
    print(f"  Sharpe Ratio   : {sh:.3f}  (annualised, hourly)")
    print(_sep())
    print(f"  Latest Signal  : {sig}")
    print(f"  Latest Regime  : {reg}")
    print(f"  Latest Votes   : {votes}/8")
    print()
    print(_sep("═"))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    # ── Build config from CLI arguments ────────────────────────────────────
    is_aggressive = args.mode == "aggressive"
    cfg = AppConfig(
        data=DataConfig(
            period=args.period,
        ),
        hmm=HMMConfig(
            n_components=args.states,
        ),
        strategy=StrategyConfig(
            mode="Aggressive" if is_aggressive else "Normal",
            min_votes=5 if is_aggressive else 7,
        ),
        backtest=BacktestConfig(
            starting_capital=args.capital,
            fee_pct=args.fee,
            slippage_pct=args.slip,
        ),
    )

    print()
    print(_sep("═"))
    print("  Bitcoin Market Regime Detection and Backtesting System")
    print(_sep("═"))
    print(f"  Mode      : {cfg.strategy.mode}  ({cfg.strategy.min_votes}/8 votes required)")
    print(f"  HMM States: {cfg.hmm.n_components}")
    print(f"  Capital   : ${cfg.backtest.starting_capital:,.0f}")
    print(f"  Fee       : {cfg.backtest.fee_pct:.2f}% per side")
    print(f"  Slippage  : {cfg.backtest.slippage_pct:.2f}% per side")
    print(f"  Period    : {cfg.data.period}  (hourly BTC-USD)")
    print(_sep("═"))

    # ── Stage 1: Download data ─────────────────────────────────────────────
    print("\n[1/5] Downloading BTC-USD hourly data from yfinance ...")
    df = fetch_data(cfg.data)
    print(f"      {len(df):,} hourly bars  |  "
          f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")

    # ── Stages 2-5: Run the full pipeline ──────────────────────────────────
    print("\n[2/5] Engineering features ...")
    print("[3/5] Training Gaussian HMM ...")
    print("[4/5] Generating strategy signals ...")
    print("[5/5] Simulating backtest ...")

    engine  = BacktestEngine(cfg)
    result  = engine.run(df)
    metrics = compute_metrics(result, df)

    # ── Print results ──────────────────────────────────────────────────────
    _print_metrics(metrics)

    # ── HMM state summary ──────────────────────────────────────────────────
    if result.regime_model is not None:
        print("  HMM State Summary (ranked by mean return):")
        print()
        summary = result.regime_model.state_summary()
        print(summary.to_string(index=False))
        print()

    # ── Export outputs ─────────────────────────────────────────────────────
    equity_path = OUTPUTS_DIR / "equity_curve.csv"
    result.equity_curve.to_csv(equity_path)
    print(f"  Equity curve saved  → {equity_path}")

    if result.trades:
        trade_log_path = OUTPUTS_DIR / "trade_log.csv"
        trades_to_dataframe(result.trades).to_csv(trade_log_path, index=False)
        print(f"  Trade log saved     → {trade_log_path}")

    print()


if __name__ == "__main__":
    main()
