"""CLI for multi-timeframe confluence.

Run forecasts on the same symbol at multiple timeframes (each with its
own CSV / fetcher / horizon) and print a confluence vote.

Example:

    python -m signals.confluence_cli \\
        --symbol ES \\
        --tf "1m:csv:data/es_1m.csv:60" \\
        --tf "5m:csv:data/es_5m.csv:60" \\
        --tf "15m:csv:data/es_15m.csv:30"

Each --tf argument is "label:fetcher_spec:horizon".
"""
from __future__ import annotations

import argparse
import sys
from typing import List

import pandas as pd

from .confluence import format_confluence_report, multi_timeframe_vote
from .fetchers import make_fetcher
from .stats import compute_stats


def _parse_tf(spec: str):
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise SystemExit(
            f"--tf must be 'label:fetcher_spec:horizon', got {spec!r}"
        )
    label, fetcher_spec, horizon = parts
    return label, fetcher_spec, int(horizon)


def main(argv=None):
    p = argparse.ArgumentParser(description="Kronos multi-timeframe confluence")
    p.add_argument("--symbol", default="UNKNOWN")
    p.add_argument(
        "--tf", action="append", required=True,
        help="format: 'label:fetcher_spec:horizon' (e.g. '5m:csv:foo.csv:60'). "
             "Can be repeated.",
    )
    p.add_argument("--lookback", type=int, default=400)
    p.add_argument("--samples", type=int, default=80)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--max-context", type=int, default=512)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    tf_specs = [_parse_tf(s) for s in args.tf]

    print(f"Loading {args.tokenizer} and {args.model} ...", file=sys.stderr)
    from model import Kronos, KronosTokenizer

    from .paths import PathForecaster

    tok = KronosTokenizer.from_pretrained(args.tokenizer)
    mdl = Kronos.from_pretrained(args.model)
    forecaster = PathForecaster(
        mdl, tok, device=args.device, max_context=args.max_context
    )

    timeframe_stats: List = []
    for label, fetcher_spec, horizon in tf_specs:
        fetcher = make_fetcher(fetcher_spec)
        df = fetcher.fetch_bars(args.symbol, label, args.lookback + 50)
        if len(df) < args.lookback:
            raise SystemExit(
                f"[{label}] only {len(df)} bars; need {args.lookback}"
            )
        hist = df.iloc[-args.lookback :].reset_index(drop=True)
        x_ts = hist["timestamps"]
        diffs = x_ts.diff().dropna()
        interval = diffs.median()
        y_ts = pd.Series([x_ts.iloc[-1] + interval * (i + 1) for i in range(horizon)])

        feature_cols = [
            c for c in ["open", "high", "low", "close", "volume", "amount"]
            if c in hist.columns
        ]
        print(
            f"[{label}] sampling {args.samples} paths over {horizon} bars ...",
            file=sys.stderr,
        )
        paths = forecaster.forecast_paths(
            df=hist[feature_cols],
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=horizon,
            T=args.temperature,
            top_p=args.top_p,
            sample_count=args.samples,
            verbose=False,
        )
        stats = compute_stats(paths, hist)
        timeframe_stats.append((label, stats))

    result = multi_timeframe_vote(timeframe_stats)
    print(format_confluence_report(result))


if __name__ == "__main__":
    main()
