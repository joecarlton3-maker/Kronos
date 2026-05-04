"""Live watch loop: re-run a forecast at intervals and alert on material change.

The loop logic and the change-detection logic are separated so the
classifier (`detect_material_change`) can be unit-tested without sleeps
or live data.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

from .stats import PathStats


@dataclass
class ChangeThresholds:
    p_up_delta: float = 0.10        # absolute change in P(up_terminal)
    confidence_delta: float = 1.0   # change in dispersion-in-ATR units
    new_high_touch_prob: float = 0.7  # any level crossing this rate triggers alert


@dataclass
class MaterialChange:
    reasons: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.reasons)


def _bias_label_from_p(p: float) -> str:
    if p >= 0.55:
        return "LONG"
    if p <= 0.45:
        return "SHORT"
    return "NEUTRAL"


def detect_material_change(
    prev: Optional[PathStats],
    current: PathStats,
    thresholds: Optional[ChangeThresholds] = None,
) -> MaterialChange:
    """Compare two consecutive forecasts; return reasons if material change."""
    th = thresholds or ChangeThresholds()
    change = MaterialChange()
    if prev is None:
        change.reasons.append("first forecast in session")
        return change

    if abs(current.p_up_terminal - prev.p_up_terminal) >= th.p_up_delta:
        change.reasons.append(
            f"P(up) shifted {prev.p_up_terminal:.0%} -> {current.p_up_terminal:.0%}"
        )

    prev_bias = _bias_label_from_p(prev.p_up_terminal)
    curr_bias = _bias_label_from_p(current.p_up_terminal)
    if prev_bias != curr_bias:
        change.reasons.append(f"bias flipped {prev_bias} -> {curr_bias}")

    if (
        prev.dispersion_in_atr == prev.dispersion_in_atr  # not NaN
        and current.dispersion_in_atr == current.dispersion_in_atr
        and abs(current.dispersion_in_atr - prev.dispersion_in_atr) >= th.confidence_delta
    ):
        change.reasons.append(
            f"dispersion shift {prev.dispersion_in_atr:.2f} -> {current.dispersion_in_atr:.2f} ATR"
        )

    prev_levels = {(t.label, t.level): t.prob for t in prev.level_touches}
    for t in current.level_touches:
        key = (t.label, t.level)
        prev_p = prev_levels.get(key, 0.0)
        if t.prob >= th.new_high_touch_prob > prev_p:
            label = t.label or f"{t.level:.2f}"
            change.reasons.append(
                f"level {label} touch prob crossed threshold ({prev_p:.0%} -> {t.prob:.0%})"
            )

    return change


def watch_loop(
    forecast_fn: Callable[[], PathStats],
    *,
    interval_seconds: float,
    alert_fn: Optional[Callable[[MaterialChange, PathStats], None]] = None,
    thresholds: Optional[ChangeThresholds] = None,
    max_iterations: Optional[int] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Run `forecast_fn` every `interval_seconds`, printing alerts on
    material change. `max_iterations` is for testing; None means run forever.
    """
    alert_fn = alert_fn or _default_alert
    last: Optional[PathStats] = None
    i = 0
    while max_iterations is None or i < max_iterations:
        try:
            stats = forecast_fn()
        except Exception as exc:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] forecast error: {exc}",
                  file=sys.stderr)
            sleep_fn(interval_seconds)
            i += 1
            continue
        change = detect_material_change(last, stats, thresholds)
        if change:
            alert_fn(change, stats)
        last = stats
        i += 1
        if max_iterations is None or i < max_iterations:
            sleep_fn(interval_seconds)


def _default_alert(change: MaterialChange, stats: PathStats) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] ALERT: P(up)={stats.p_up_terminal:.0%}", file=sys.stderr)
    for reason in change.reasons:
        print(f"  * {reason}", file=sys.stderr)
