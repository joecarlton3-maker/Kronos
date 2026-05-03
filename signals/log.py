"""Forecast log: persist per-forecast records to JSONL for later calibration.

Each line is one forecast record. Outcomes are filled in either by the
backtest harness (immediately) or by `signals.fill_outcomes` (when the
horizon for a live forecast has fully elapsed).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from .stats import PathStats


@dataclass
class Outcome:
    actual_terminal_close: float
    actual_max_high: float
    actual_min_low: float
    direction_up: bool
    hit_target_long: bool
    hit_stop_long: bool
    long_first_touch: str  # "target" / "stop" / "neither"
    realized_r_long: float
    hit_target_short: bool
    hit_stop_short: bool
    short_first_touch: str
    realized_r_short: float
    level_actual_touches: List[bool] = field(default_factory=list)


@dataclass
class ForecastRecord:
    ts_generated: str
    ts_history_end: str
    ts_horizon_end: str
    symbol: str
    timeframe: str
    lookback: int
    horizon: int
    samples: int
    stats: dict
    per_path_terminal_close: List[float]
    per_path_max_high: List[float]
    per_path_min_low: List[float]
    outcome: Optional[Outcome] = None


def record_from_paths(
    paths: np.ndarray,
    stats: PathStats,
    *,
    ts_generated: datetime,
    ts_history_end: datetime,
    ts_horizon_end: datetime,
    symbol: str,
    timeframe: str,
    lookback: int,
) -> ForecastRecord:
    return ForecastRecord(
        ts_generated=ts_generated.isoformat(),
        ts_history_end=ts_history_end.isoformat(),
        ts_horizon_end=ts_horizon_end.isoformat(),
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        horizon=stats.horizon_bars,
        samples=stats.sample_count,
        stats=asdict(stats),
        per_path_terminal_close=paths[:, -1, 3].astype(float).tolist(),
        per_path_max_high=paths[:, :, 1].max(axis=1).astype(float).tolist(),
        per_path_min_low=paths[:, :, 2].min(axis=1).astype(float).tolist(),
    )


def write_record(path: Path, rec: ForecastRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(rec), default=str) + "\n")


def iter_records(path: Path) -> Iterator[ForecastRecord]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            outcome_d = d.pop("outcome", None)
            outcome = Outcome(**outcome_d) if outcome_d else None
            yield ForecastRecord(outcome=outcome, **d)
