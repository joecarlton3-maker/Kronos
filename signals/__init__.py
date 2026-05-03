"""Probabilistic decision-support layer on top of Kronos forecasts.

`PathForecaster` is imported lazily so that `signals.stats` / `signals.report`
can be used without torch (e.g. when post-processing pre-computed paths).
"""
from .levels import auto_key_levels
from .plan import PlanEvaluation, evaluate_plan
from .report import format_report
from .stats import LevelTouch, PathStats, compute_stats
from .trajectory import (
    DrawdownStats,
    TimeToTouch,
    TrajectoryBand,
    drawdown_long,
    drawdown_short,
    per_bar_band,
    time_to_touch,
)

__all__ = [
    "PathForecaster",
    "PathStats",
    "LevelTouch",
    "compute_stats",
    "format_report",
    "auto_key_levels",
    "PlanEvaluation",
    "evaluate_plan",
    "TrajectoryBand",
    "DrawdownStats",
    "TimeToTouch",
    "per_bar_band",
    "drawdown_long",
    "drawdown_short",
    "time_to_touch",
]


def __getattr__(name):
    if name == "PathForecaster":
        from .paths import PathForecaster

        return PathForecaster
    raise AttributeError(f"module 'signals' has no attribute {name!r}")
