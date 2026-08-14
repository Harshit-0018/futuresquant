"""
FuturesQuant — Systematic Futures Trading Backtester
------------------------------------------------------
A lightweight backtesting engine for momentum-based trading strategies on
futures/commodities price series (e.g. soybean complex, crude oil, treasury
note futures, SOFR-linked instruments).

Design goals:
  - Works on ANY OHLC price series you feed it (CSV or DataFrame) — plug in
    real data from a broker/data vendor feed (e.g. yfinance tickers like
    ZS=F soybean, ZN=F 10Y note futures, CL=F crude) once you have network
    access to a market data source.
  - Ships with a synthetic Geometric Brownian Motion generator so the whole
    pipeline runs end-to-end with zero external dependencies, for demo /
    unit-testing purposes.
  - Computes standard risk/performance metrics used to evaluate a trading
    strategy: CAGR, annualized Sharpe ratio, max drawdown, win rate.

Usage:
    python backtester.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# 1. Data layer
# --------------------------------------------------------------------------
def generate_synthetic_futures_series(
    n_days: int = 750,
    start_price: float = 1350.0,
    annual_drift: float = 0.05,
    annual_vol: float = 0.22,
    seed: int = 7,
    name: str = "SYNTH_SOYBEAN",
) -> pd.DataFrame:
    """Simulate a daily futures price series via Geometric Brownian Motion
    with mild autocorrelated regime shifts, to mimic commodity trend/chop
    cycles. Swap this out for real data (CSV or a broker/data-vendor feed)
    when available.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    # Inject 3-4 regime blocks with different drift so the series has
    # genuine trends for a momentum strategy to catch (not pure noise).
    n_regimes = 4
    regime_len = n_days // n_regimes
    drifts = rng.uniform(-0.15, 0.25, size=n_regimes)

    log_returns = []
    for i in range(n_regimes):
        mu = drifts[i]
        length = regime_len if i < n_regimes - 1 else n_days - regime_len * (n_regimes - 1)
        shocks = rng.normal(
            (mu - 0.5 * annual_vol ** 2) * dt, annual_vol * np.sqrt(dt), size=length
        )
        log_returns.append(shocks)
    log_returns = np.concatenate(log_returns)

    prices = start_price * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    df = pd.DataFrame({"date": dates, "close": prices})
    df["open"] = df["close"].shift(1).fillna(start_price) * (1 + rng.normal(0, 0.001, n_days))
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + rng.uniform(0, 0.004, n_days))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - rng.uniform(0, 0.004, n_days))
    df.attrs["name"] = name
    return df.set_index("date")


def load_price_csv(path: str, price_col: str = "close") -> pd.DataFrame:
    """Load a real OHLC series from CSV. Expected columns: date, open,
    high, low, close (case-insensitive) — see data/sample_data_template.csv
    for the exact format. Use this once you have a real data source (broker
    export, data vendor, or a licensed market data API).
    """
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


# --------------------------------------------------------------------------
# 2. Technical indicators
# --------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame, fast: int = 20, slow: int = 60, rsi_period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(fast).mean()
    df["sma_slow"] = df["close"].rolling(slow).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20-day, 2 std)
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    return df


# --------------------------------------------------------------------------
# 3. Strategy: SMA crossover momentum with RSI filter
# --------------------------------------------------------------------------
def generate_signals(df: pd.DataFrame, rsi_overbought: float = 75, rsi_oversold: float = 25) -> pd.DataFrame:
    """Long when fast SMA > slow SMA and RSI is not overbought.
    Flat/short when fast SMA < slow SMA and RSI is not oversold.
    """
    df = df.copy()
    long_cond = (df["sma_fast"] > df["sma_slow"]) & (df["rsi"] < rsi_overbought)
    short_cond = (df["sma_fast"] < df["sma_slow"]) & (df["rsi"] > rsi_oversold)

    df["position"] = 0
    df.loc[long_cond, "position"] = 1
    df.loc[short_cond, "position"] = -1
    df["position"] = df["position"].ffill().fillna(0)
    return df


# --------------------------------------------------------------------------
# 4. Backtest engine + performance metrics
# --------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, txn_cost_bps: float = 2.0) -> pd.DataFrame:
    """Vectorized backtest: apply yesterday's position to today's return,
    subtract transaction costs on position changes (in basis points).
    """
    df = df.copy()
    df["asset_return"] = df["close"].pct_change()
    df["position_lag"] = df["position"].shift(1).fillna(0)
    df["turnover"] = df["position_lag"].diff().abs().fillna(0)
    df["txn_cost"] = df["turnover"] * (txn_cost_bps / 10_000)
    df["strategy_return"] = df["position_lag"] * df["asset_return"] - df["txn_cost"]
    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()
    df["buy_hold_curve"] = (1 + df["asset_return"]).cumprod()
    return df


def performance_summary(df: pd.DataFrame, periods_per_year: int = 252) -> dict:
    strat = df["strategy_return"].dropna()
    n_years = len(strat) / periods_per_year

    cagr = df["equity_curve"].iloc[-1] ** (1 / n_years) - 1
    sharpe = (strat.mean() / strat.std()) * np.sqrt(periods_per_year) if strat.std() > 0 else np.nan

    running_max = df["equity_curve"].cummax()
    drawdown = df["equity_curve"] / running_max - 1
    max_dd = drawdown.min()

    trades = strat[df["turnover"].reindex(strat.index).fillna(0) > 0]
    daily_pos_returns = strat[strat != 0]
    win_rate = (daily_pos_returns > 0).mean() if len(daily_pos_returns) else np.nan

    bh_cagr = df["buy_hold_curve"].iloc[-1] ** (1 / n_years) - 1

    return {
        "CAGR (strategy)": f"{cagr:.2%}",
        "CAGR (buy & hold)": f"{bh_cagr:.2%}",
        "Annualized Sharpe": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Win rate (daily, in-position)": f"{win_rate:.2%}",
        "Total trades (position flips)": int((df['turnover'] > 0).sum()),
    }


# --------------------------------------------------------------------------
# 5. Plotting
# --------------------------------------------------------------------------
def plot_results(df: pd.DataFrame, asset_name: str, out_path: str = "equity_curve.png"):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(df.index, df["close"], label=f"{asset_name} price", color="#333333", linewidth=1)
    axes[0].plot(df.index, df["sma_fast"], label="SMA fast", linewidth=1)
    axes[0].plot(df.index, df["sma_slow"], label="SMA slow", linewidth=1)
    axes[0].set_title(f"{asset_name} — Price & SMA Crossover Signal")
    axes[0].legend(loc="upper left")

    axes[1].plot(df.index, df["equity_curve"], label="Strategy equity curve", linewidth=1.5)
    axes[1].plot(df.index, df["buy_hold_curve"], label="Buy & hold", linewidth=1, linestyle="--")
    axes[1].set_title("Cumulative Growth of $1")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    asset = "Synthetic Soybean Futures (demo)"
    prices = generate_synthetic_futures_series(name=asset)
    prices = add_indicators(prices)
    prices = generate_signals(prices)
    results = run_backtest(prices)

    summary = performance_summary(results)
    print(f"\n=== Backtest Summary: {asset} ===")
    for k, v in summary.items():
        print(f"{k:35s}: {v}")

    plot_results(results, asset)
