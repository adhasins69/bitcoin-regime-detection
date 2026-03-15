"""
app.py
======
Streamlit dashboard for the Bitcoin Market Regime Detection and Backtesting System.

Displays:
  - Sidebar: mode, HMM states, capital, fee, slippage, run button
  - Metrics row: signal, regime, total return, alpha, win rate, drawdown, trades
  - Charts: candlestick + EMA overlays + regime shading, equity curve, drawdown
  - Trade log table
  - HMM state summary expander

Run
---
    streamlit run app.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config — MUST be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BTC Regime System",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Local imports (after set_page_config)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    AppConfig, DataConfig, HMMConfig, StrategyConfig, BacktestConfig,
)
from data_loader import fetch_data
from regime_model import BULL, BEAR, NEUTRAL, REGIME_COLORS
from backtester import (
    BacktestEngine, BacktestResult,
    compute_metrics, trades_to_dataframe,
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.metric-card {
    background: #1c2333;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    border: 1px solid #2d3748;
}
.metric-label { font-size: 0.76rem; color: #8892a4; margin-bottom: 5px; }
.metric-value { font-size: 1.75rem; font-weight: 700; line-height: 1.15; }
.metric-sub   { font-size: 0.70rem; color: #8892a4; margin-top: 4px; }
div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Sidebar
# ===========================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("### Strategy Mode")
    mode_choice = st.radio(
        "Mode",
        options=["Normal", "Aggressive"],
        index=0,
        help=(
            "**Normal** — requires 7 out of 8 confirmation conditions\n\n"
            "**Aggressive** — requires only 5 out of 8 confirmation conditions"
        ),
    )
    is_aggressive = mode_choice == "Aggressive"

    if is_aggressive:
        st.warning("⚡ Aggressive mode — lower entry threshold (5/8 votes)")

    st.markdown("---")
    st.markdown("### Model Parameters")

    n_states = st.slider(
        "HMM Hidden States", min_value=3, max_value=9, value=5, step=1,
        help="Number of latent market regimes. Default 5.",
    )

    st.markdown("---")
    st.markdown("### Capital & Costs")

    starting_capital = st.number_input(
        "Starting Capital (USD)",
        min_value=100.0, max_value=1_000_000.0,
        value=10_000.0, step=1_000.0, format="%.0f",
    )
    fee_pct = st.slider(
        "Exchange Fee (% per side)",
        min_value=0.0, max_value=0.5, value=0.10, step=0.01, format="%.2f",
        help="e.g. Binance maker fee = 0.10%",
    )
    slippage_pct = st.slider(
        "Slippage (% per side)",
        min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.2f",
        help="Estimated price impact on entry and exit",
    )

    st.markdown("---")
    run_btn = st.button("🚀 Run Backtest", use_container_width=True, type="primary")
    st.markdown("---")

    votes_str = "5 / 8" if is_aggressive else "7 / 8"
    st.markdown("**Active Parameters**")
    st.markdown(f"""
| Parameter     | Value        |
|---------------|--------------|
| Mode          | {mode_choice} |
| HMM States    | {n_states}   |
| Min Votes     | {votes_str}  |
| Fee           | {fee_pct:.2f}% |
| Slippage      | {slippage_pct:.2f}% |
""")
    st.markdown("---")
    st.markdown("**8 Confirmation Conditions**")
    st.markdown("""
1. RSI < 90
2. Momentum > 1%
3. Volatility < 6%
4. Volume > 20-bar SMA
5. ADX > 25
6. Close > EMA 50
7. Close > EMA 200
8. MACD > Signal Line
""")
    st.caption("Data: BTC-USD · 1h · 730 days (yfinance)")


# ===========================================================================
# Session state initialisation
# ===========================================================================

for _k in ("result", "metrics", "df_raw"):
    if _k not in st.session_state:
        st.session_state[_k] = None


# ===========================================================================
# Config builder
# ===========================================================================

def _build_config() -> AppConfig:
    return AppConfig(
        data=DataConfig(),
        hmm=HMMConfig(n_components=n_states),
        strategy=StrategyConfig(
            mode=mode_choice,
            min_votes=5 if is_aggressive else 7,
        ),
        backtest=BacktestConfig(
            starting_capital=starting_capital,
            fee_pct=fee_pct,
            slippage_pct=slippage_pct,
        ),
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_cached(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Cache BTC data for 1 hour — avoids re-downloading on parameter changes."""
    return fetch_data(DataConfig(ticker=ticker, period=period, interval=interval))


# ===========================================================================
# Chart builders
# ===========================================================================

def _regime_shapes(regimes: pd.Series) -> list[dict]:
    """
    Build Plotly shape rectangles for regime background shading.
    Green = Bull, Red = Bear, Gray = Neutral.
    """
    shapes: list[dict] = []
    if regimes.empty:
        return shapes

    prev  = regimes.iloc[0]
    start = regimes.index[0]

    def _add(x0, x1, regime: str):
        color = REGIME_COLORS.get(regime, "rgba(180,180,180,0.08)")
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=str(x0), x1=str(x1), y0=0, y1=1,
            fillcolor=color, opacity=1.0, layer="below", line_width=0,
        ))

    for ts, reg in regimes.items():
        if reg != prev:
            _add(start, ts, prev)
            start = ts
            prev  = reg
    _add(start, regimes.index[-1], prev)

    return shapes


def _entry_exit_traces(
    trades, cutoff: pd.Timestamp
) -> tuple[go.Scatter | None, go.Scatter | None]:
    """Build Plotly scatter traces for trade entry and exit markers."""
    visible = [t for t in trades if t.entry_time >= cutoff]
    if not visible:
        return None, None

    entry_x, entry_y, entry_text = [], [], []
    exit_x,  exit_y,  exit_text  = [], [], []

    for t in visible:
        entry_x.append(t.entry_time)
        entry_y.append(t.raw_entry_price)
        entry_text.append(
            f"ENTRY {t.entry_time.strftime('%Y-%m-%d %H:%M')}<br>"
            f"Price: ${t.raw_entry_price:,.0f}   Votes: {t.votes_at_entry}/8"
        )
        exit_x.append(t.exit_time)
        exit_y.append(t.raw_exit_price)
        exit_text.append(
            f"EXIT ({t.exit_reason}) {t.exit_time.strftime('%Y-%m-%d %H:%M')}<br>"
            f"Price: ${t.raw_exit_price:,.0f}   Return: {t.return_pct:+.2f}%"
        )

    entry_trace = go.Scatter(
        x=entry_x, y=entry_y, mode="markers", name="Entry",
        marker=dict(symbol="triangle-up", size=12, color="#00e676",
                    line=dict(width=1, color="#fff")),
        hovertext=entry_text, hoverinfo="text",
    )
    exit_trace = go.Scatter(
        x=exit_x, y=exit_y, mode="markers", name="Exit",
        marker=dict(symbol="triangle-down", size=12, color="#ef5350",
                    line=dict(width=1, color="#fff")),
        hovertext=exit_text, hoverinfo="text",
    )
    return entry_trace, exit_trace


def build_candle_chart(result: BacktestResult, chart_days: int = 180) -> go.Figure:
    """
    Candlestick chart with:
    - EMA 50 and EMA 200 overlays
    - Regime background shading (Bull=green, Bear=red, Neutral=gray)
    - Entry and exit trade markers
    - Volume bars in a sub-panel
    """
    df      = result.signal_df
    cutoff  = df.index[-1] - pd.Timedelta(days=chart_days)
    plot_df = df[df.index >= cutoff].copy()
    regimes = plot_df["regime"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.80, 0.20],
    )

    # ── Candlestick ───────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"],   close=plot_df["Close"],
        name="BTC-USD",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # ── EMA overlays ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["EMA_50"],
        name="EMA 50", line=dict(color="#ffa726", width=1.2, dash="dot"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["EMA_200"],
        name="EMA 200", line=dict(color="#ce93d8", width=1.5, dash="dash"),
    ), row=1, col=1)

    # ── Trade markers ─────────────────────────────────────────────────────
    en_tr, ex_tr = _entry_exit_traces(result.trades, cutoff)
    if en_tr:
        fig.add_trace(en_tr, row=1, col=1)
    if ex_tr:
        fig.add_trace(ex_tr, row=1, col=1)

    # ── Volume bars ───────────────────────────────────────────────────────
    vol_colors = np.where(
        plot_df["Close"] >= plot_df["Open"], "#26a69a", "#ef5350"
    )
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df["Volume"],
        name="Volume", marker_color=vol_colors, opacity=0.55,
    ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        shapes=_regime_shapes(regimes),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", family="Inter, Arial, sans-serif"),
        margin=dict(l=6, r=6, t=6, b=6), height=580,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor="#1e2737", showgrid=True,
        ),
        yaxis=dict(gridcolor="#1e2737", tickprefix="$"),
        yaxis2=dict(gridcolor="#1e2737"),
        hovermode="x unified",
    )
    return fig


def build_equity_chart(result: BacktestResult, df_raw: pd.DataFrame) -> go.Figure:
    """Line chart comparing strategy equity curve vs buy-and-hold."""
    port = result.equity_curve["value"]
    sc   = result.config.backtest.starting_capital
    bh   = df_raw["Close"].reindex(result.equity_curve.index, method="ffill")
    bh   = bh / bh.iloc[0] * sc

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=port.index, y=port.values,
        name=f"Strategy ({result.config.strategy.mode})",
        line=dict(color="#29b6f6", width=2),
        fill="tozeroy", fillcolor="rgba(41,182,246,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=bh.index, y=bh.values,
        name="Buy & Hold",
        line=dict(color="#ff7043", width=1.5, dash="dot"),
    ))
    fig.add_hline(
        y=sc, line_dash="dash", line_color="rgba(255,255,255,0.18)",
        annotation_text=f"${sc:,.0f} start",
        annotation_font_color="rgba(255,255,255,0.35)",
    )

    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        margin=dict(l=6, r=6, t=6, b=6), height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0),
        xaxis=dict(gridcolor="#1e2737"),
        yaxis=dict(gridcolor="#1e2737", tickprefix="$"),
        hovermode="x unified",
    )
    return fig


def build_drawdown_chart(result: BacktestResult) -> go.Figure:
    """Area chart showing portfolio drawdown from peak."""
    port = result.equity_curve["value"]
    dd   = (port - port.cummax()) / port.cummax() * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name="Drawdown",
        line=dict(color="#ef5350", width=1.2),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)")

    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        margin=dict(l=6, r=6, t=6, b=6), height=230,
        xaxis=dict(gridcolor="#1e2737"),
        yaxis=dict(gridcolor="#1e2737", ticksuffix="%"),
        hovermode="x unified",
    )
    return fig


# ===========================================================================
# Metric card helper
# ===========================================================================

def _card(label: str, value: str, color: str = "#e0e0e0", sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color};">{value}</div>'
        f'{sub_html}</div>'
    )


# ===========================================================================
# Main page layout
# ===========================================================================

def main():
    # ── Page header ───────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;color:#29b6f6;margin-bottom:2px;'>"
        "₿ Bitcoin Market Regime Detection & Backtesting</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#8892a4;font-size:0.83rem;'>"
        f"Gaussian HMM · {n_states} Hidden States · 8-Condition Vote System · "
        f"{mode_choice} Mode · Long-Only · No Leverage</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Run backtest ──────────────────────────────────────────────────────
    if run_btn:
        cfg = _build_config()
        with st.spinner(
            "Downloading BTC-USD data and training HMM ... "
            "(first run takes ~20–40 s)"
        ):
            try:
                df = _fetch_cached(
                    cfg.data.ticker, cfg.data.period, cfg.data.interval
                )
                engine  = BacktestEngine(cfg)
                result  = engine.run(df)
                metrics = compute_metrics(result, df)

                st.session_state.result  = result
                st.session_state.metrics = metrics
                st.session_state.df_raw  = df

            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                return

    result:  BacktestResult | None = st.session_state.result
    metrics: dict | None           = st.session_state.metrics
    df_raw:  pd.DataFrame | None   = st.session_state.df_raw

    # ── Welcome screen ────────────────────────────────────────────────────
    if result is None:
        st.markdown("""
<div style='text-align:center;padding:50px 20px;color:#8892a4;'>
  <h3 style='color:#29b6f6;'>Configure and run your first backtest</h3>
  <p style='font-size:1.0rem;'>
    Use the sidebar to set parameters, then click <b>🚀 Run Backtest</b>.
  </p>
  <br>
  <p>
    The system will:<br><br>
    • Download 730 days of BTC-USD 1-hour candles from yfinance<br>
    • Compute 12 technical and statistical features<br>
    • Train a Gaussian HMM to detect market regimes (Bull / Bear / Neutral)<br>
    • Apply an 8-condition confirmation vote filter<br>
    • Simulate a long-only strategy with realistic fee and slippage<br>
    • Display interactive charts, equity curve, and performance metrics
  </p>
</div>
""", unsafe_allow_html=True)
        return

    # ── Metrics row 1: core results ───────────────────────────────────────
    sig   = metrics["current_signal"]
    reg   = metrics["current_regime"]
    ret   = metrics["total_return_pct"]
    bh    = metrics["bh_return_pct"]
    alpha = metrics["alpha"]
    wr    = metrics["win_rate"]
    dd    = metrics["max_drawdown"]
    n     = metrics["num_trades"]
    sh    = metrics["sharpe"]
    fv    = metrics["final_value"]
    votes = metrics["current_votes"]
    mv    = metrics["min_votes"]

    sig_color = "#00e676" if "LONG" in sig else "#ffa726"
    reg_color = (
        "#00e676" if reg == BULL
        else "#ef5350" if reg == BEAR
        else "#90a4ae"
    )
    rc = lambda v: "#00e676" if v >= 0 else "#ef5350"

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    with c1:
        st.markdown(_card("Latest Signal",    sig,              sig_color), unsafe_allow_html=True)
    with c2:
        st.markdown(_card("Latest Regime",    reg,              reg_color), unsafe_allow_html=True)
    with c3:
        st.markdown(_card("Total Return",     f"{ret:+.2f}%",   rc(ret),
                          sub=f"B&H: {bh:+.1f}%"),             unsafe_allow_html=True)
    with c4:
        st.markdown(_card("Alpha vs B&H",     f"{alpha:+.2f}%", rc(alpha)), unsafe_allow_html=True)
    with c5:
        st.markdown(_card("Win Rate",         f"{wr:.1f}%",
                          "#00e676" if wr >= 50 else "#ef5350",
                          sub=f"{n} trades"),                   unsafe_allow_html=True)
    with c6:
        st.markdown(_card("Max Drawdown",     f"{dd:.2f}%",     "#ef5350",
                          sub=f"Sharpe: {sh:.2f}"),             unsafe_allow_html=True)
    with c7:
        bar = "█" * votes + "░" * (8 - votes)
        vc  = "#00e676" if votes >= mv else ("#ffa726" if votes >= mv - 2 else "#ef5350")
        st.markdown(_card(f"Votes ({mv}/8 req)", f"{votes}/8",  vc,
                          sub=bar),                             unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics row 2: portfolio details ─────────────────────────────────
    d1, d2, d3, d4 = st.columns(4)
    sc = metrics["start_capital"]
    with d1:
        st.markdown(_card("Final Portfolio",  f"${fv:,.0f}",    "#29b6f6",
                          sub=f"Started: ${sc:,.0f}"),          unsafe_allow_html=True)
    with d2:
        bh_final = df_raw["Close"].iloc[-1] / df_raw["Close"].iloc[0] * sc
        st.markdown(_card("B&H Portfolio",    f"${bh_final:,.0f}", "#ff7043",
                          sub=f"Buy & hold from ${sc:,.0f}"),   unsafe_allow_html=True)
    with d3:
        data_start = df_raw.index[0].strftime("%Y-%m-%d")
        data_end   = df_raw.index[-1].strftime("%Y-%m-%d")
        st.markdown(_card("Data Range",       data_end,         "#e0e0e0",
                          sub=f"From {data_start}"),            unsafe_allow_html=True)
    with d4:
        n_bars = len(result.signal_df)
        st.markdown(_card("Hourly Bars",      f"{n_bars:,}",    "#e0e0e0",
                          sub="BTC-USD · 1h"),                  unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Candlestick chart ─────────────────────────────────────────────────
    st.subheader("BTC-USD Hourly — Regime Detection Overlay")

    chart_days = st.slider(
        "Chart window (days)",
        min_value=30, max_value=730, value=180, step=30,
    )
    st.plotly_chart(build_candle_chart(result, chart_days), use_container_width=True)
    st.markdown(
        "<p style='font-size:0.78rem;color:#8892a4;'>"
        "🟢 Green = Bull &nbsp;·&nbsp; 🔴 Red = Bear &nbsp;·&nbsp; "
        "⬜ Gray = Neutral &nbsp;·&nbsp; "
        "▲ Entry &nbsp;·&nbsp; ▼ Exit &nbsp;·&nbsp; "
        "Orange dashed = EMA 50 &nbsp;·&nbsp; Purple dashed = EMA 200"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Equity curve and drawdown ─────────────────────────────────────────
    st.subheader("Equity Curve & Drawdown")

    eq_col, dd_col = st.columns([3, 2])
    with eq_col:
        st.caption("Strategy portfolio vs Buy & Hold (both starting at selected capital)")
        st.plotly_chart(build_equity_chart(result, df_raw), use_container_width=True)
    with dd_col:
        st.caption("Portfolio drawdown from running peak")
        st.plotly_chart(build_drawdown_chart(result), use_container_width=True)

    # ── Trade log ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"Trade Log  ({len(result.trades)} trades)")

    if result.trades:
        trade_df = trades_to_dataframe(result.trades)

        # Display-friendly formatting
        display = pd.DataFrame({
            "Entry Time":      trade_df["entry_time"].dt.strftime("%Y-%m-%d %H:%M"),
            "Exit Time":       trade_df["exit_time"].dt.strftime("%Y-%m-%d %H:%M"),
            "Entry Price":     trade_df["entry_price"].apply(lambda x: f"${x:,.2f}"),
            "Exit Price":      trade_df["exit_price"].apply(lambda x: f"${x:,.2f}"),
            "PnL ($)":         trade_df["pnl"].apply(lambda x: f"${x:+,.2f}"),
            "Return (%)":      trade_df["return_pct"].apply(lambda x: f"{x:+.3f}%"),
            "Exit Reason":     trade_df["exit_reason"],
            "Votes at Entry":  trade_df["votes_at_entry"].apply(lambda x: f"{x}/8"),
        })

        def _style_pnl(val: str) -> str:
            return "color: #00e676" if "+" in val else "color: #ef5350"

        styled = display.style.applymap(_style_pnl, subset=["PnL ($)", "Return (%)"])
        st.dataframe(styled, use_container_width=True, height=320)
    else:
        st.info(
            "No trades were taken in this backtest window.  "
            "Try Aggressive mode or a smaller HMM state count."
        )

    # ── HMM state summary ─────────────────────────────────────────────────
    with st.expander(
        f"🧠 HMM State Summary ({result.config.hmm.n_components} hidden states)",
        expanded=False,
    ):
        if result.regime_model is not None:
            summary = result.regime_model.state_summary()
            st.dataframe(summary, use_container_width=True)
            st.caption(
                "Sorted by Mean_Return (descending).  "
                "Bull = highest mean return state.  Bear = lowest mean return state."
            )
        else:
            st.info("Regime model not available.")

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;font-size:0.75rem;color:#5a6070;'>"
        "Bitcoin Market Regime Detection &amp; Backtesting System · "
        "Gaussian HMM · Python 3.11 · "
        "For educational and research purposes only · Not financial advice"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
