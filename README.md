# Bitcoin Market Regime Detection and Backtesting System

A Python portfolio project demonstrating **Hidden Markov Model (HMM) regime detection**, **technical feature engineering**, and **event-driven backtesting** on 2 years of hourly BTC-USD data.

Built for: statistics students and junior quants targeting analytics, data science, and finance roles.

---

## Overview

This system uses a **Gaussian Hidden Markov Model** to identify latent market regimes in Bitcoin price data (Bull, Bear, Neutral), then combines the detected regime with **8 technical confirmation filters** to generate a long-only trading strategy. The strategy is backtested against a buy-and-hold benchmark with realistic fee and slippage assumptions.

### Pipeline

```
yfinance (BTC-USD 1h)
       │
       ▼
Feature Engineering
  ├── Returns, Range, Vol_Change   ← HMM training features
  └── RSI, Momentum, Volatility, ADX, EMA50/200, MACD  ← vote filters
       │
       ▼
Gaussian HMM (hmmlearn)
  ├── 5 hidden states
  ├── Bull  = state with highest mean return
  ├── Bear  = state with lowest mean return
  └── Neutral = all remaining states
       │
       ▼
Strategy Signal Generation
  ├── Base rule:  LONG only when regime == Bull
  └── Confirmation: 7/8 conditions must pass (Normal) or 5/8 (Aggressive)
       │
       ▼
Event-Driven Backtester
  ├── Long-only, no leverage
  ├── Fee + slippage applied symmetrically
  ├── Exit on signal-off or Bear regime
  └── Metrics: return, alpha, Sharpe, drawdown, win rate
       │
       ▼
Streamlit Dashboard  (app.py)
  └── Candlestick chart · Regime shading · Equity curve · Trade log
```

---

## Features

| Module | Responsibility |
|--------|---------------|
| `config.py` | Single source of truth — all parameters in one place |
| `data_loader.py` | yfinance download with MultiIndex handling and cleaning |
| `features.py` | 12 indicators implemented from scratch (no TA library needed) |
| `regime_model.py` | GaussianHMM wrapper with automatic Bull/Bear state identification |
| `strategy.py` | 8-condition vote system, Normal and Aggressive modes |
| `backtester.py` | Event-driven simulation, metrics, trade log |
| `main.py` | CLI entry point — runs full pipeline headlessly |
| `app.py` | Interactive Streamlit dashboard |

### Confirmation Conditions

| # | Condition | Rationale |
|---|-----------|-----------|
| 1 | RSI < 90 | Not overbought |
| 2 | Momentum > 1% | Positive price momentum |
| 3 | Volatility < 6% | Manageable annualised vol |
| 4 | Volume > 20-bar SMA | Above-average participation |
| 5 | ADX > 25 | Trend is strong |
| 6 | Close > EMA 50 | Short-term trend aligned |
| 7 | Close > EMA 200 | Long-term trend aligned |
| 8 | MACD > Signal Line | Momentum accelerating |

---

## Quickstart (Windows PowerShell)

### 1. Create and activate virtual environment

```powershell
# Navigate to project folder
cd "C:\Users\adhaz\OneDrive\Desktop\trading bot\trade 1"

# Create Python 3.11 virtual environment
py -3.11 -m venv venv

# Activate
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the backtest (CLI)

```powershell
python main.py
```

With options:

```powershell
python main.py --mode aggressive      # Aggressive mode (5/8 votes)
python main.py --states 7             # 7 HMM states
python main.py --capital 50000        # Custom starting capital
python main.py --fee 0.05             # Lower fee
python main.py --help                 # All options
```

### 4. Launch the Streamlit dashboard

```powershell
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## Project Structure

```
trade-1/
├── config.py          # All tunable parameters
├── data_loader.py     # yfinance data download and cleaning
├── features.py        # Feature engineering (12 indicators)
├── regime_model.py    # Gaussian HMM regime detection
├── strategy.py        # Signal generation with vote system
├── backtester.py      # Event-driven simulation and metrics
├── main.py            # CLI entry point
├── app.py             # Streamlit dashboard
├── requirements.txt   # Pinned dependencies
├── outputs/           # CSV exports (equity curve, trade log)
└── figures/           # Chart exports (optional)
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| yfinance | 0.2.51 | Market data download |
| hmmlearn | 0.3.2 | Gaussian HMM |
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical computation |
| scikit-learn | 1.5.1 | StandardScaler |
| streamlit | 1.36.0 | Dashboard |
| plotly | 5.22.0 | Interactive charts |
| scipy | 1.13.1 | Statistical utilities |
| matplotlib | 3.9.1 | Static chart export |

---

## Strategy Modes

| Mode | Min Votes | Use Case |
|------|-----------|----------|
| Normal | 7 / 8 | Conservative — fewer trades, higher quality signals |
| Aggressive | 5 / 8 | More trades — accepts weaker confirmation |

---

## Backtesting Assumptions

- **Long-only** — no shorts, no leverage
- **Full capital deployment** per trade
- **Entry cost**: Close × (1 + fee + slippage)
- **Exit proceeds**: Close × (1 − fee + slippage)
- **Exit triggers**: signal turns CASH *or* regime turns Bear
- **End-of-data**: any open position closed at the final bar

---

## Outputs

After running `main.py`, two CSV files are saved to `outputs/`:

- `equity_curve.csv` — per-bar portfolio value, regime, and position flag
- `trade_log.csv` — entry/exit times, prices, PnL, and return per trade

---

## Troubleshooting (Windows)

**1. `py -3.11 -m venv venv` fails — Python 3.11 not found**
Install Python 3.11 from [python.org](https://www.python.org/downloads/) and check "Add to PATH". Then use `py -3.11` or `python3.11`.

**2. `.\venv\Scripts\Activate.ps1` — execution policy error**
Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**3. `hmmlearn` install fails — no C compiler**
Install Visual Studio Build Tools from [visualstudio.microsoft.com/downloads](https://visualstudio.microsoft.com/downloads/) (select "C++ build tools" workload), or use a pre-built wheel:
```powershell
pip install hmmlearn --only-binary=:all:
```

**4. yfinance returns empty or MultiIndex DataFrame**
Update yfinance: `pip install --upgrade yfinance`. The `data_loader.py` module handles both old and new column formats.

**5. Streamlit app is slow on first load**
The first run downloads ~17,000 hourly bars and trains the HMM — this takes 20–40 seconds. Subsequent runs within the same session use Streamlit's 1-hour cache (`@st.cache_data(ttl=3600)`).

---

## Disclaimer

This project is for **educational and portfolio purposes only**. It does not constitute financial advice. Backtesting results do not guarantee future performance. All code is provided as-is with no warranty.

---

## License

MIT
