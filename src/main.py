from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INITIAL_CAPITAL = 100000.0
ENTRY_SLIPPAGE = 0.0005
EXIT_SLIPPAGE = 0.0005
LOOKBACK_ENTRY = 240  # 10 days of 1h bars
LOOKBACK_STOP = 72  # 3 days of 1h bars
ANNUALIZATION_FACTOR = np.sqrt(24 * 365)


@dataclass
class Position:
    side: int  # 1 for long, -1 for short
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    qty: float
    entry_equity: float


@dataclass
class PendingOrder:
    kind: str  # "entry" or "exit"
    side: int
    execute_i: int


def generate_synthetic_ohlcv(
    start: str = "2021-01-01",
    end: str = "2026-04-01",
    freq: str = "1h",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    n = len(index)

    drift = 0.00003
    vol = 0.012
    log_returns = drift + vol * rng.standard_normal(n)
    close = 1200 * np.exp(np.cumsum(log_returns))

    open_noise = rng.normal(0.0, 0.0015, n)
    open_ = np.concatenate(([close[0]], close[:-1])) * (1 + open_noise)

    hi_noise = np.abs(rng.normal(0.0015, 0.001, n))
    lo_noise = np.abs(rng.normal(0.0015, 0.001, n))
    high = np.maximum(open_, close) * (1 + hi_noise)
    low = np.minimum(open_, close) * (1 - lo_noise)
    low = np.maximum(low, 1e-6)

    volume = rng.lognormal(mean=8.0, sigma=0.5, size=n)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    return df


def load_ohlcv_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index(ts)
    elif "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index(ts)
    else:
        raise ValueError("CSV must include 'timestamp' or 'datetime' column.")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    out = df[["open", "high", "low", "close", "volume"]].copy()
    out = out.sort_index()
    return out


def get_data(
    data_path: Optional[Path],
    start: str = "2021-01-01",
    end: str = "2026-04-01",
) -> pd.DataFrame:
    if data_path is not None and data_path.exists():
        data = load_ohlcv_from_csv(data_path)
        data = data.loc[(data.index >= pd.Timestamp(start, tz="UTC")) & (data.index <= pd.Timestamp(end, tz="UTC"))]
        if data.empty:
            raise ValueError("Loaded CSV data is empty for the requested date range.")
        return data
    return generate_synthetic_ohlcv(start=start, end=end)


def apply_entry_slippage(next_open: float, side: int) -> float:
    if side == 1:
        return next_open * (1 + ENTRY_SLIPPAGE)
    return next_open * (1 - ENTRY_SLIPPAGE)


def apply_exit_slippage(next_open: float, side: int) -> float:
    if side == 1:
        return next_open * (1 - EXIT_SLIPPAGE)
    return next_open * (1 + EXIT_SLIPPAGE)


def compute_max_drawdown(equity: pd.Series) -> float:
    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1.0
    return float(drawdown.min())


def run_backtest(data: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> tuple[pd.Series, pd.DataFrame, dict]:
    n = len(data)
    idx = data.index

    capital = initial_capital
    position: Optional[Position] = None
    pending_entry: Optional[PendingOrder] = None
    pending_exit: Optional[PendingOrder] = None

    equity_values: list[float] = []
    trade_records: list[dict] = []

    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    opens = data["open"].to_numpy()
    closes = data["close"].to_numpy()

    for i in range(n):
        ts = idx[i]
        exited_this_bar = False

        if pending_exit is not None and pending_exit.execute_i == i and position is not None:
            next_open = opens[i]
            exit_price = apply_exit_slippage(next_open, position.side)
            pnl = position.side * position.qty * (exit_price - position.entry_price)
            capital = position.entry_equity + pnl

            trade_records.append(
                {
                    "entry_time": position.entry_time,
                    "exit_time": ts,
                    "side": "LONG" if position.side == 1 else "SHORT",
                    "entry_price": position.entry_price,
                    "exit_price": exit_price,
                    "stop_price": position.stop_price,
                    "qty": position.qty,
                    "pnl": pnl,
                    "return_pct": pnl / position.entry_equity,
                    "exit_reason": "stop",
                }
            )

            position = None
            pending_exit = None
            exited_this_bar = True

        if pending_entry is not None and pending_entry.execute_i == i and position is None and not exited_this_bar:
            side = pending_entry.side
            next_open = opens[i]
            entry_price = apply_entry_slippage(next_open, side)

            if i < LOOKBACK_STOP:
                pending_entry = None
            else:
                if side == 1:
                    stop_price = float(np.min(lows[i - LOOKBACK_STOP : i]))
                else:
                    stop_price = float(np.max(highs[i - LOOKBACK_STOP : i]))

                qty = capital / entry_price
                position = Position(
                    side=side,
                    entry_time=ts,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    qty=qty,
                    entry_equity=capital,
                )
                pending_entry = None

        if position is not None:
            mtm_equity = position.entry_equity + position.side * position.qty * (closes[i] - position.entry_price)
            equity_values.append(mtm_equity)
        else:
            equity_values.append(capital)

        if i == n - 1:
            continue

        if position is not None:
            if pending_exit is None:
                if position.side == 1 and lows[i] <= position.stop_price:
                    pending_exit = PendingOrder(kind="exit", side=position.side, execute_i=i + 1)
                elif position.side == -1 and highs[i] >= position.stop_price:
                    pending_exit = PendingOrder(kind="exit", side=position.side, execute_i=i + 1)
            continue

        if exited_this_bar:
            continue

        if pending_entry is not None:
            continue

        if i < LOOKBACK_ENTRY:
            continue

        prev_high = float(np.max(highs[i - LOOKBACK_ENTRY : i]))
        prev_low = float(np.min(lows[i - LOOKBACK_ENTRY : i]))

        long_signal = highs[i] > prev_high
        short_signal = lows[i] < prev_low

        if long_signal and not short_signal:
            pending_entry = PendingOrder(kind="entry", side=1, execute_i=i + 1)
        elif short_signal and not long_signal:
            pending_entry = PendingOrder(kind="entry", side=-1, execute_i=i + 1)

    equity_curve = pd.Series(equity_values, index=idx, name="equity")

    trade_log = pd.DataFrame(trade_records)
    if not trade_log.empty:
        trade_log["entry_time"] = pd.to_datetime(trade_log["entry_time"], utc=True)
        trade_log["exit_time"] = pd.to_datetime(trade_log["exit_time"], utc=True)

    returns = equity_curve.pct_change().fillna(0.0)
    total_return = equity_curve.iloc[-1] / initial_capital - 1.0

    total_hours = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 3600
    total_years = total_hours / (24 * 365)
    annualized_return = (equity_curve.iloc[-1] / initial_capital) ** (1 / total_years) - 1 if total_years > 0 else np.nan

    annualized_vol = returns.std(ddof=0) * ANNUALIZATION_FACTOR
    sharpe = (returns.mean() / returns.std(ddof=0) * ANNUALIZATION_FACTOR) if returns.std(ddof=0) > 0 else np.nan
    max_drawdown = compute_max_drawdown(equity_curve)

    num_trades = len(trade_log)
    win_rate = float((trade_log["pnl"] > 0).mean()) if num_trades > 0 else np.nan

    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "number_of_trades": num_trades,
        "win_rate": win_rate,
    }

    return equity_curve, trade_log, metrics


def save_outputs(equity_curve: pd.Series, trade_log: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    trade_log_path = output_dir / "trade_log.csv"
    trade_log.to_csv(trade_log_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve.index, equity_curve.values, label="Equity Curve")
    plt.title("Equity Curve")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Equity (USDT)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.close()


def print_metrics(metrics: dict) -> None:
    print("Backtest Performance")
    print("-" * 24)
    print(f"Total Return:         {metrics['total_return']:.2%}")
    print(f"Annualized Return:    {metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility:{metrics['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown:     {metrics['max_drawdown']:.2%}")
    print(f"Number of Trades:     {metrics['number_of_trades']}")
    if np.isnan(metrics["win_rate"]):
        print("Win Rate:             N/A")
    else:
        print(f"Win Rate:             {metrics['win_rate']:.2%}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "ethusdt_1h.csv"

    data = get_data(data_path=data_path)
    equity_curve, trade_log, metrics = run_backtest(data)

    save_outputs(equity_curve, trade_log, project_root / "reports")
    print_metrics(metrics)
    print(f"Trade log saved to: {(project_root / 'reports' / 'trade_log.csv')}")
    print(f"Equity curve saved to: {(project_root / 'reports' / 'equity_curve.png')}")


if __name__ == "__main__":
    main()
