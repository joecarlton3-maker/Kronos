"""Per-bar trajectory bands, drawdown distribution, and time-to-touch
helpers — the building blocks for "when does the move happen?" analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class TrajectoryBand:
    """Per-bar percentiles. Each list has length = pred_len."""
    p10: List[float]
    p25: List[float]
    p50: List[float]
    p75: List[float]
    p90: List[float]


@dataclass
class DrawdownStats:
    """Distribution of max adverse excursion (in price points) for paths
    held to horizon from `last_price`. Always non-negative."""
    median: float
    p25: float          # 75% of paths see at least this much heat
    p75: float          # 25% of paths see worse than this
    p90: float          # tail risk
    max_observed: float


@dataclass
class TimeToTouch:
    """Distribution of bars until first touch of a price level."""
    label: str
    price: float
    direction: str             # 'above' or 'below'
    p_touched: float
    median_bars: Optional[float]    # None if p_touched < 0.5
    p25_bars: Optional[float]
    p75_bars: Optional[float]
    p_touched_first_quarter: float
    p_touched_first_half: float


def per_bar_band(values: np.ndarray) -> TrajectoryBand:
    """`values`: (n_samples, pred_len). Returns per-bar percentile lists."""
    pcts = np.percentile(values, [10, 25, 50, 75, 90], axis=0)
    return TrajectoryBand(
        p10=pcts[0].tolist(),
        p25=pcts[1].tolist(),
        p50=pcts[2].tolist(),
        p75=pcts[3].tolist(),
        p90=pcts[4].tolist(),
    )


def drawdown_long(paths: np.ndarray, last_price: float) -> DrawdownStats:
    lows = paths[:, :, 2]
    dd = np.maximum(last_price - lows.min(axis=1), 0.0)
    return _drawdown_stats(dd)


def drawdown_short(paths: np.ndarray, last_price: float) -> DrawdownStats:
    highs = paths[:, :, 1]
    dd = np.maximum(highs.max(axis=1) - last_price, 0.0)
    return _drawdown_stats(dd)


def _drawdown_stats(dd: np.ndarray) -> DrawdownStats:
    return DrawdownStats(
        median=float(np.percentile(dd, 50)),
        p25=float(np.percentile(dd, 25)),
        p75=float(np.percentile(dd, 75)),
        p90=float(np.percentile(dd, 90)),
        max_observed=float(dd.max()),
    )


def time_to_touch(
    paths: np.ndarray,
    level: float,
    last_price: float,
    label: str = "",
) -> TimeToTouch:
    """First-touch distribution for a price level across all paths."""
    n, T, _ = paths.shape
    highs = paths[:, :, 1]
    lows = paths[:, :, 2]
    if level >= last_price:
        touched_mask = highs >= level
        direction = "above"
    else:
        touched_mask = lows <= level
        direction = "below"

    any_touch = touched_mask.any(axis=1)
    p_touched = float(any_touch.mean())

    # First-touch bar index (1-based), only for paths that touched
    first_idx = touched_mask.argmax(axis=1)  # zero if never touched
    first_bars = first_idx[any_touch] + 1  # 1-based
    if first_bars.size > 0:
        median_bars = float(np.percentile(first_bars, 50))
        p25_bars = float(np.percentile(first_bars, 25))
        p75_bars = float(np.percentile(first_bars, 75))
    else:
        median_bars = p25_bars = p75_bars = None

    quarter = max(1, T // 4)
    half = max(1, T // 2)
    p_first_quarter = float(touched_mask[:, :quarter].any(axis=1).mean())
    p_first_half = float(touched_mask[:, :half].any(axis=1).mean())

    return TimeToTouch(
        label=label,
        price=float(level),
        direction=direction,
        p_touched=p_touched,
        median_bars=median_bars,
        p25_bars=p25_bars,
        p75_bars=p75_bars,
        p_touched_first_quarter=p_first_quarter,
        p_touched_first_half=p_first_half,
    )
