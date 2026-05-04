"""Expected-R calculator for a user-supplied trade plan.

Walks every forecast path bar by bar to determine first-touch of stop or
target (conservative: stop wins same-bar ties), then summarises:
  - P(target first), P(stop first), P(neither)
  - Expected R (full reward if target hit, -1R if stop, terminal close if neither)
  - Distribution of bars-to-resolution
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PlanEvaluation:
    side: str                  # 'long' or 'short'
    entry: float
    stop: float
    target: float
    risk_per_unit: float       # |entry - stop|
    reward_per_unit: float     # |target - entry|
    rr_planned: float

    p_target_first: float
    p_stop_first: float
    p_neither: float

    expected_r: float          # blended expectancy
    avg_r_when_target: float
    avg_r_when_neither: float

    median_bars_to_target: Optional[float]
    median_bars_to_stop: Optional[float]
    median_bars_to_resolution: Optional[float]


def evaluate_plan(
    paths: np.ndarray,
    *,
    entry: float,
    stop: float,
    target: float,
    side: str = "long",
) -> PlanEvaluation:
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    if paths.ndim != 3 or paths.shape[2] < 4:
        raise ValueError("paths must have shape (n_samples, pred_len, >=4)")

    if side == "long":
        if not (stop < entry < target):
            raise ValueError("for long: must have stop < entry < target")
        risk = entry - stop
        reward = target - entry
    else:
        if not (target < entry < stop):
            raise ValueError("for short: must have target < entry < stop")
        risk = stop - entry
        reward = entry - target

    n, T, _ = paths.shape
    highs = paths[:, :, 1]
    lows = paths[:, :, 2]
    closes = paths[:, :, 3]

    if side == "long":
        hit_stop_mask = lows <= stop
        hit_target_mask = highs >= target
    else:
        hit_stop_mask = highs >= stop
        hit_target_mask = lows <= target

    SENTINEL = T + 1
    first_stop = np.where(
        hit_stop_mask.any(axis=1), hit_stop_mask.argmax(axis=1) + 1, SENTINEL
    )
    first_target = np.where(
        hit_target_mask.any(axis=1), hit_target_mask.argmax(axis=1) + 1, SENTINEL
    )

    # Conservative: stop wins same-bar ties
    stop_first = first_stop <= first_target
    target_first = first_target < first_stop
    neither = (first_stop == SENTINEL) & (first_target == SENTINEL)
    # Override stop_first / target_first with neither where applicable
    stop_first = stop_first & ~neither
    target_first = target_first & ~neither

    p_target = float(target_first.mean())
    p_stop = float(stop_first.mean())
    p_neither = float(neither.mean())

    # Realized R per path
    realized_r = np.zeros(n)
    realized_r[target_first] = reward / risk
    realized_r[stop_first] = -1.0
    if neither.any():
        terminal = closes[neither, -1]
        if side == "long":
            realized_r[neither] = (terminal - entry) / risk
        else:
            realized_r[neither] = (entry - terminal) / risk

    expected_r = float(realized_r.mean())
    avg_target = (
        float(realized_r[target_first].mean()) if target_first.any() else float("nan")
    )
    avg_neither = (
        float(realized_r[neither].mean()) if neither.any() else float("nan")
    )

    def _med(arr):
        return float(np.median(arr)) if arr.size > 0 else None

    median_bars_target = _med(first_target[target_first])
    median_bars_stop = _med(first_stop[stop_first])
    resolved_bars = np.where(
        target_first, first_target, np.where(stop_first, first_stop, SENTINEL)
    )
    resolved = resolved_bars[resolved_bars < SENTINEL]
    median_bars_resolution = _med(resolved)

    return PlanEvaluation(
        side=side,
        entry=float(entry),
        stop=float(stop),
        target=float(target),
        risk_per_unit=float(risk),
        reward_per_unit=float(reward),
        rr_planned=float(reward / risk),
        p_target_first=p_target,
        p_stop_first=p_stop,
        p_neither=p_neither,
        expected_r=expected_r,
        avg_r_when_target=avg_target,
        avg_r_when_neither=avg_neither,
        median_bars_to_target=median_bars_target,
        median_bars_to_stop=median_bars_stop,
        median_bars_to_resolution=median_bars_resolution,
    )
