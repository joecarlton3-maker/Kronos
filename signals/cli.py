"""CLI for the Kronos probabilistic forecast dashboard.

Usage (run from repo root):

    python -m signals.cli \\
        --csv data/es_5m.csv \\
        --symbol ES --timeframe 5m \\
        --lookback 400 --horizon 60 --samples 100 \\
        --auto-levels \\
        --levels 5160,5180,5120,5100 \\
        --plan side=long,entry=5142,stop=5128,target=5180

The CSV must contain columns: timestamps, open, high, low, close.
volume and amount are optional.

Flags of note:
    --auto-levels           Compute VWAP, Today/Prior-Day H/L, Prior Close
                            and add as labeled levels.
    --levels                Extra comma-separated raw price levels.
    --plan                  Evaluate a user-supplied trade plan against the
                            forecast paths. Format:
                              "side=long,entry=5142,stop=5128,target=5180"
    --json                  Emit the full stats blob (and plan, if given) as
                            JSON to stdout instead of the text report.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .fetchers import CSVBarFetcher, BarFetcher, make_fetcher
from .levels import auto_key_levels
from .log import record_from_paths, write_record
from .plan import PlanEvaluation, evaluate_plan
from .report import format_report
from .stats import compute_stats
from .watch import ChangeThresholds, watch_loop


def _infer_bar_interval(ts: pd.Series) -> pd.Timedelta:
    diffs = ts.diff().dropna()
    if diffs.empty:
        raise ValueError("Need >=2 timestamps to infer bar interval.")
    return diffs.median()


def _future_timestamps(
    last_ts: pd.Timestamp, interval: pd.Timedelta, n: int
) -> pd.Series:
    return pd.Series([last_ts + interval * (i + 1) for i in range(n)])


def _parse_plan(spec: str) -> dict:
    """Parse 'side=long,entry=5142,stop=5128,target=5180' into a dict."""
    out = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise SystemExit(f"--plan part {part!r} missing '='")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    for required in ("side", "entry", "stop", "target"):
        if required not in out:
            raise SystemExit(f"--plan missing required key: {required}")
    return {
        "side": out["side"],
        "entry": float(out["entry"]),
        "stop": float(out["stop"]),
        "target": float(out["target"]),
    }


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if math.isnan(float(obj)):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Cannot serialize {type(obj)}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Kronos probabilistic forecast dashboard"
    )
    p.add_argument("--csv", default=None, help="OHLCV CSV with 'timestamps' column")
    p.add_argument(
        "--fetcher", default=None,
        help="Bar fetcher spec: csv:PATH, subprocess:CMD, tradingview[:CMD]. "
             "Mutually exclusive with --csv.",
    )
    p.add_argument("--symbol", default="UNKNOWN")
    p.add_argument("--timeframe", default="?")
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--horizon", type=int, default=60, help="bars to forecast")
    p.add_argument("--samples", type=int, default=100, help="number of forecast paths")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--levels", default="", help="comma-separated price levels")
    p.add_argument(
        "--auto-levels", action="store_true",
        help="auto-detect VWAP / prior-day H/L / prior close from input data",
    )
    p.add_argument(
        "--plan", default=None,
        help="evaluate a trade plan: side=long,entry=...,stop=...,target=...",
    )
    p.add_argument("--json", action="store_true", help="emit JSON instead of text report")
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
    p.add_argument(
        "--chart", default=None,
        help="write a PNG fan chart to this path",
    )
    p.add_argument(
        "--watch", type=float, default=None, metavar="SECONDS",
        help="re-run the forecast every N seconds and alert on material change",
    )
    p.add_argument(
        "--watch-iterations", type=int, default=None,
        help="cap number of watch iterations (default: run forever)",
    )
    args = p.parse_args(argv)
    if (args.csv is None) == (args.fetcher is None):
        raise SystemExit("Provide exactly one of --csv or --fetcher")

    fetcher: BarFetcher
    if args.csv:
        fetcher = CSVBarFetcher(args.csv)
    else:
        fetcher = make_fetcher(args.fetcher)

    print(f"Loading {args.tokenizer} and {args.model} ...", file=sys.stderr)
    from model import Kronos, KronosTokenizer

    from .paths import PathForecaster

    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.model)
    forecaster = PathForecaster(
        model, tokenizer, device=args.device, max_context=args.max_context
    )

    def run_once():
        df = fetcher.fetch_bars(args.symbol, args.timeframe, args.lookback + 50)
        if len(df) < args.lookback:
            raise SystemExit(
                f"Only {len(df)} bars available; need at least lookback={args.lookback}"
            )
        hist = df.iloc[-args.lookback :].reset_index(drop=True)
        x_ts = hist["timestamps"]
        interval = _infer_bar_interval(x_ts)
        y_ts = _future_timestamps(x_ts.iloc[-1], interval, args.horizon)

        levels = []
        if args.auto_levels:
            levels.extend(auto_key_levels(hist))
        for tok in args.levels.split(","):
            tok = tok.strip()
            if tok:
                levels.append(("", float(tok)))

        feature_cols = ["open", "high", "low", "close"]
        if "volume" in hist.columns:
            feature_cols.append("volume")
        if "amount" in hist.columns:
            feature_cols.append("amount")

        paths_arr = forecaster.forecast_paths(
            df=hist[feature_cols],
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=args.horizon,
            T=args.temperature,
            top_p=args.top_p,
            sample_count=args.samples,
            verbose=args.verbose,
        )
        stats = compute_stats(paths_arr, hist, levels=levels)
        plan_eval: Optional[PlanEvaluation] = None
        if args.plan:
            plan_eval = evaluate_plan(paths_arr, **_parse_plan(args.plan))

        if args.json:
            out = {
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "generated_at": datetime.now().isoformat(),
                "lookback_bars": args.lookback,
                "stats": asdict(stats),
                "plan": asdict(plan_eval) if plan_eval is not None else None,
            }
            print(json.dumps(out, default=_json_default, indent=2))
        else:
            print(format_report(
                stats,
                symbol=args.symbol,
                timeframe=args.timeframe,
                lookback_bars=args.lookback,
                plan=plan_eval,
            ))

        if args.chart:
            from .chart import fan_chart

            fan_chart(
                stats, hist, args.chart,
                symbol=args.symbol, timeframe=args.timeframe,
                plan=plan_eval, levels=levels or None,
            )
            print(f"Wrote fan chart to {args.chart}", file=sys.stderr)

        if args.log_to:
            rec = record_from_paths(
                paths_arr, stats,
                ts_generated=datetime.now(),
                ts_history_end=pd.Timestamp(x_ts.iloc[-1]).to_pydatetime(),
                ts_horizon_end=pd.Timestamp(y_ts.iloc[-1]).to_pydatetime(),
                symbol=args.symbol, timeframe=args.timeframe,
                lookback=args.lookback,
            )
            write_record(Path(args.log_to), rec)

        return stats

    if args.watch is None:
        run_once()
        return

    print(
        f"Watch mode: re-running every {args.watch}s "
        f"(material-change alerts to stderr).",
        file=sys.stderr,
    )
    watch_loop(
        run_once,
        interval_seconds=args.watch,
        thresholds=ChangeThresholds(),
        max_iterations=args.watch_iterations,
    )


if __name__ == "__main__":
    main()
