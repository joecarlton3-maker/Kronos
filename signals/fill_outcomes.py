"""Fill realized outcomes into a forecast log once enough bars exist.

For live use: `signals.cli` writes records without outcomes. After the
horizon has elapsed, run this to populate the outcome field by joining
against an updated OHLC CSV.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .backtest import _evaluate_outcome
from .log import iter_records


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Fill outcomes into a Kronos forecast log"
    )
    p.add_argument("--log", required=True, help="JSONL log to update in-place")
    p.add_argument("--csv", required=True, help="OHLC CSV with up-to-date bars")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    log_path = Path(args.log)
    records = list(iter_records(log_path))
    updated = 0
    new_lines = []
    for r in records:
        if r.outcome is not None:
            new_lines.append(json.dumps(asdict(r), default=str))
            continue
        history_end = pd.Timestamp(r.ts_history_end)
        horizon_end = pd.Timestamp(r.ts_horizon_end)
        actual_window = df[
            (df["timestamps"] > history_end) & (df["timestamps"] <= horizon_end)
        ]
        if len(actual_window) < r.horizon:
            new_lines.append(json.dumps(asdict(r), default=str))
            continue
        levels = [l["level"] for l in r.stats.get("level_touches", [])]
        outcome = _evaluate_outcome(
            actual_bars=actual_window.head(r.horizon).reset_index(drop=True),
            last_price=r.stats["last_price"],
            suggested_stop_long=r.stats["suggested_stop_long"],
            suggested_target_long=r.stats["suggested_target_long"],
            suggested_stop_short=r.stats["suggested_stop_short"],
            suggested_target_short=r.stats["suggested_target_short"],
            levels=levels,
        )
        r.outcome = outcome
        new_lines.append(json.dumps(asdict(r), default=str))
        updated += 1

    log_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
    print(f"Updated {updated} records in {log_path}")


if __name__ == "__main__":
    main()
