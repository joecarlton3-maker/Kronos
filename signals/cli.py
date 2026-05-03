"""CLI for the Kronos probabilistic forecast dashboard.

Usage (run from repo root):

    python -m signals.cli \\
        --csv data/es_5m.csv \\
        --symbol ES --timeframe 5m \\
        --lookback 400 --horizon 60 --samples 100 \\
        --levels 5160,5180,5120,5100

The CSV must contain columns: timestamps, open, high, low, close.
volume and amount are optional.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from model import Kronos, KronosTokenizer

from .log import record_from_paths, write_record
from .paths import PathForecaster
from .report import format_report
from .stats import compute_stats


def _infer_bar_interval(ts: pd.Series) -> pd.Timedelta:
    diffs = ts.diff().dropna()
    if diffs.empty:
        raise ValueError("Need >=2 timestamps to infer bar interval.")
    return diffs.median()


def _future_timestamps(
    last_ts: pd.Timestamp, interval: pd.Timedelta, n: int
) -> pd.Series:
    return pd.Series([last_ts + interval * (i + 1) for i in range(n)])


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Kronos probabilistic forecast dashboard"
    )
    p.add_argument("--csv", required=True, help="OHLCV CSV with 'timestamps' column")
    p.add_argument("--symbol", default="UNKNOWN")
    p.add_argument("--timeframe", default="?")
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--horizon", type=int, default=60, help="bars to forecast")
    p.add_argument("--samples", type=int, default=100, help="number of forecast paths")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--levels", default="", help="comma-separated price levels")
    p.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--max-context", type=int, default=512)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--log-to",
        default=None,
        help="append this forecast (without outcome) to a JSONL log "
             "for later calibration (see signals.fill_outcomes)",
    )
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    if "timestamps" not in df.columns:
        raise SystemExit("CSV must have a 'timestamps' column")
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    if len(df) < args.lookback:
        raise SystemExit(
            f"Only {len(df)} rows in CSV; need at least lookback={args.lookback}"
        )

    hist = df.iloc[-args.lookback :].reset_index(drop=True)
    x_ts = hist["timestamps"]
    interval = _infer_bar_interval(x_ts)
    y_ts = _future_timestamps(x_ts.iloc[-1], interval, args.horizon)

    levels = [float(x) for x in args.levels.split(",") if x.strip()]

    print(f"Loading {args.tokenizer} and {args.model} ...", file=sys.stderr)
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    forecaster = PathForecaster(
        model, tokenizer, device=args.device, max_context=args.max_context
    )

    feature_cols = ["open", "high", "low", "close"]
    if "volume" in hist.columns:
        feature_cols.append("volume")
    if "amount" in hist.columns:
        feature_cols.append("amount")

    print(
        f"Sampling {args.samples} paths over {args.horizon} bars ...",
        file=sys.stderr,
    )
    paths = forecaster.forecast_paths(
        df=hist[feature_cols],
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=args.horizon,
        T=args.temperature,
        top_p=args.top_p,
        sample_count=args.samples,
        verbose=args.verbose,
    )

    stats = compute_stats(paths, hist, levels=levels)
    report = format_report(
        stats,
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_bars=args.lookback,
    )
    print(report)

    if args.log_to:
        rec = record_from_paths(
            paths,
            stats,
            ts_generated=datetime.now(),
            ts_history_end=pd.Timestamp(x_ts.iloc[-1]).to_pydatetime(),
            ts_horizon_end=pd.Timestamp(y_ts.iloc[-1]).to_pydatetime(),
            symbol=args.symbol,
            timeframe=args.timeframe,
            lookback=args.lookback,
        )
        write_record(Path(args.log_to), rec)
        print(f"Logged forecast to {args.log_to}", file=sys.stderr)


if __name__ == "__main__":
    main()
