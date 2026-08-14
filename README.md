# FuturesQuant — Systematic Futures Trading Backtester

A lightweight, dependency-light backtesting engine for evaluating momentum
trading strategies on futures/commodities price series (e.g. soybean
complex, crude oil, treasury note futures).

## What it does
- **Signal generation**: SMA(20/60) crossover with an RSI(14) overbought/
  oversold filter to avoid entering trend trades at extremes.
- **Vectorized backtest**: applies lagged positions to daily returns,
  charges transaction costs (bps) on every position flip.
- **Performance metrics**: CAGR, annualized Sharpe ratio, max drawdown,
  win rate, trade count — vs. a buy-and-hold benchmark.
- **Plots**: price + moving averages, and cumulative equity curve.

## Why
Built to practice the quantitative workflow behind systematic trade
execution — signal design, risk-adjusted performance measurement, and
transaction-cost-aware backtesting — the same core skills used to develop
and evaluate strategies in commodities and fixed income markets.

## Data
Ships with a synthetic Geometric Brownian Motion price generator (with
regime shifts, so there are real trends to trade) for a zero-dependency
demo. Swap in real data by pointing `load_price_csv()` at a CSV with
`date, open, high, low, close` columns — e.g. an export from a market
data vendor or broker feed.

## Project structure
```
futuresquant/
├── backtester.py                  # data layer, indicators, strategy, backtest engine, plots
├── requirements.txt
├── LICENSE
├── data/
│   └── sample_data_template.csv   # expected CSV format for real OHLC data
└── sample_output/
    └── equity_curve.png           # example run on the built-in synthetic series
```

## Run it
```bash
git clone <your-repo-url>
cd futuresquant
pip install -r requirements.txt
python backtester.py
```
This runs the full pipeline on the built-in synthetic series and writes
`equity_curve.png` to the project root.

To use real data instead, format a CSV like `data/sample_data_template.csv`
and call:
```python
prices = load_price_csv("data/your_data.csv")
```
in place of `generate_synthetic_futures_series()` in `backtester.py`.

## Sample output
![Equity curve](sample_output/equity_curve.png)

## Sample output (synthetic demo series)
```
CAGR (strategy)                    : -1.39%
CAGR (buy & hold)                  : -23.77%
Annualized Sharpe                  : 0.02
Max Drawdown                       : -30.14%
Win rate (daily, in-position)      : 49.12%
Total trades (position flips)      : 58
```
The strategy meaningfully outperforms buy-and-hold on this trending/
mean-reverting synthetic series by cutting exposure during the sustained
downtrend — the point of the project is the framework (signal → backtest →
risk metrics), not this specific parameter set.

## Next steps
- Add walk-forward parameter optimization instead of fixed SMA windows.
- Add position sizing via volatility targeting (e.g. scale size inversely
  to realized vol) instead of fixed +1/-1 exposure.
- Backtest across multiple correlated futures (e.g. full soybean complex:
  soybean, soybean meal, soybean oil) to study crush-margin spread trades.
