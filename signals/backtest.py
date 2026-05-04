"""Walk-forward backtest harness for Kronos forecasts.

Generates forecasts at successive historical points, computes the
realized outcome over the horizon, and writes both to a JSONL log for
later calibration analysis.

The forecaster is injected so this module is testable without loading
Kronos. The CLI wires in a real `PathForecaster`.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from .log import Outcome, record_from_paths, write_record
from .stats import _normalize_levels, compute_stats


ForecasterFn = Callable[
    [pd.DataFrame, pd.Series, pd.Series, int, int], np.ndarray
]


def _evaluate_outcome(
    actual_bars: pd.DataFrame,
    last_price: float,
    suggested_stop_long: float,
    suggested_target_long: float,
    suggested_stop_short: float,
    suggested_target_short: float,
    levels: List[float],
) -> Outcome:
    closes = actual_bars["close"].values
    highs = actual_bars["high"].values
    lows = actual_bars["low"].values

    actual_terminal_close = float(closes[-1])
    actual_max_high = float(highs.max())
    actual_min_low = float(lows.min())
    direction_up = actual_terminal_close > last_price

    long_first = "neither"
    bars_to_stop_long = None
    bars_to_target_long = None
    for i, (h, l) in enumerate(zip(highs, lows), start=1):
        if bars_to_stop_long is None and l <= suggested_stop_long:
            bars_to_stop_long = i
            if long_first == "neither":
                long_first = "stop"
        if bars_to_target_long is None and h >= suggested_target_long:
            bars_to_target_long = i
            if long_first == "neither":
                long_first = "target"
        if bars_to_stop_long is not None and bars_to_target_long is not None:
            break

    short_first = "neither"
    bars_to_stop_short = None
    bars_to_target_short = None
    for i, (h, l) in enumerate(zip(highs, lows), start=1):
        if bars_to_stop_short is None and h >= suggested_stop_short:
            bars_to_stop_short = i
            if short_first == "neither":
                short_first = "stop"
        if bars_to_target_short is None and l <= suggested_target_short:
            bars_to_target_short = i
            if short_first == "neither":
                short_first = "target"
        if bars_to_stop_short is not None and bars_to_target_short is not None:
            break

    risk_long = max(last_price - suggested_stop_long, 1e-9)
    risk_short = max(suggested_stop_short - last_price, 1e-9)
    if long_first == "target":
        r_long = (suggested_target_long - last_price) / risk_long
    elif long_first == "stop":
        r_long = -1.0
    else:
        r_long = (actual_terminal_close - last_price) / risk_long
    if short_first == "target":
        r_short = (last_price - suggested_target_short) / risk_short
    elif short_first == "stop":
        r_short = -1.0
    else:
        r_short = (last_price - actual_terminal_close) / risk_short

    level_actual_touches: List[bool] = []
    level_bars: List = []
    for lvl in levels:
        if lvl > last_price:
            touched = bool(actual_max_high >= lvl)
            first_bar = None
            if touched:
                for i, h in enumerate(highs, start=1):
                    if h >= lvl:
                        first_bar = i
                        break
        else:
            touched = bool(actual_min_low <= lvl)
            first_bar = None
            if touched:
                for i, l in enumerate(lows, start=1):
                    if l <= lvl:
                        first_bar = i
                        break
        level_actual_touches.append(touched)
        level_bars.append(first_bar)

    realized_dd_long = max(last_price - actual_min_low, 0.0)
    realized_dd_short = max(actual_max_high - last_price, 0.0)

    return Outcome(
        actual_terminal_close=actual_terminal_close,
        actual_max_high=actual_max_high,
        actual_min_low=actual_min_low,
        direction_up=bool(direction_up),
        hit_target_long=long_first == "target",
        hit_stop_long=long_first == "stop",
        long_first_touch=long_first,
        realized_r_long=float(r_long),
        hit_target_short=short_first == "target",
        hit_stop_short=short_first == "stop",
        short_first_touch=short_first,
        realized_r_short=float(r_short),
        level_actual_touches=level_actual_touches,
        realized_dd_long=float(realized_dd_long),
        realized_dd_short=float(realized_dd_short),
        bars_to_target_long=bars_to_target_long,
        bars_to_stop_long=bars_to_stop_long,
        bars_to_target_short=bars_to_target_short,
        bars_to_stop_short=bars_to_stop_short,
        level_actual_bars_to_touch=level_bars,
    )


def run_backtest(
    df: pd.DataFrame,
    *,
    forecaster_fn: ForecasterFn,
    lookback: int,
    horizon: int,
    samples: int,
    step: Optional[int] = None,
    levels: Optional[List[float]] = None,
    symbol: str = "UNKNOWN",
    timeframe: str = "?",
    log_path: Optional[Path] = None,
    progress: bool = False,
) -> int:
    """Walk forward through df generating forecasts and outcomes.

    Returns the number of records written.
    """
    if "timestamps" not in df.columns:
        raise ValueError("df must have a 'timestamps' column")
    df = df.sort_values("timestamps").reset_index(drop=True)
    levels = levels or []
    step = step or horizon

    n = len(df)
    last_start = n - horizon
    starts = list(range(lookback, last_start + 1, step))
    written = 0

    iterator = starts
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(starts, desc="backtest")
        except ImportError:
            pass

    for t in iterator:
        hist = df.iloc[t - lookback : t].reset_index(drop=True)
        actual = df.iloc[t : t + horizon].reset_index(drop=True)
        if len(actual) < horizon:
            break
        x_ts = hist["timestamps"]
        y_ts = actual["timestamps"]
        feature_cols = [
            c
            for c in ["open", "high", "low", "close", "volume", "amount"]
            if c in hist.columns
        ]

        paths = forecaster_fn(hist[feature_cols], x_ts, y_ts, horizon, samples)
        stats = compute_stats(paths, hist, levels=levels)

        last_price = float(hist["close"].iloc[-1])
        plain_levels = [price for _, price in _normalize_levels(levels)]
        outcome = _evaluate_outcome(
            actual_bars=actual,
            last_price=last_price,
            suggested_stop_long=stats.suggested_stop_long,
            suggested_target_long=stats.suggested_target_long,
            suggested_stop_short=stats.suggested_stop_short,
            suggested_target_short=stats.suggested_target_short,
            levels=plain_levels,
        )

        rec = record_from_paths(
            paths,
            stats,
            ts_generated=datetime.now(),
            ts_history_end=pd.Timestamp(x_ts.iloc[-1]).to_pydatetime(),
            ts_horizon_end=pd.Timestamp(y_ts.iloc[-1]).to_pydatetime(),
            symbol=symbol,
            timeframe=timeframe,
            lookback=lookback,
        )
        rec.outcome = outcome

        if log_path is not None:
            write_record(log_path, rec)
        written += 1

    return written


def main(argv=None):
    p = argparse.ArgumentParser(description="Kronos walk-forward backtest")
    p.add_argument("--csv", required=True)
    p.add_argument("--symbol", default="UNKNOWN")
    p.add_argument("--timeframe", default="?")
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--step", type=int, default=None,
                   help="default = horizon (non-overlapping outcomes)")
    p.add_argument("--levels", default="")
    p.add_argument("--log", required=True, help="output JSONL log path")
    p.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--max-context", type=int, default=512)
    p.add_argument("--device", default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    args = p.parse_args(argv)

    from model import Kronos, KronosTokenizer

    from .paths import PathForecaster

    df = pd.read_csv(args.csv)
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    print(f"Loading {args.tokenizer} and {args.model} ...", file=sys.stderr)
    tok = KronosTokenizer.from_pretrained(args.tokenizer)
    mdl = Kronos.from_pretrained(args.model)
    forecaster = PathForecaster(
        mdl, tok, device=args.device, max_context=args.max_context
    )

    def fn(hist, x_ts, y_ts, pred_len, n_samples):
        return forecaster.forecast_paths(
            df=hist,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_len,
            T=args.temperature,
            top_p=args.top_p,
            sample_count=n_samples,
            verbose=False,
        )

    levels = [float(x) for x in args.levels.split(",") if x.strip()]
    n = run_backtest(
        df,
        forecaster_fn=fn,
        lookback=args.lookback,
        horizon=args.horizon,
        samples=args.samples,
        step=args.step,
        levels=levels,
        symbol=args.symbol,
        timeframe=args.timeframe,
        log_path=Path(args.log),
        progress=True,
    )
    print(f"Wrote {n} records to {args.log}")


if __name__ == "__main__":
    main()
