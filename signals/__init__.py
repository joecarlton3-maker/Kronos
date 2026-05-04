"""Probabilistic decision-support layer on top of Kronos forecasts.

`PathForecaster` is imported lazily so that `signals.stats` / `signals.report`
can be used without torch (e.g. when post-processing pre-computed paths).
"""
from .confluence import (
    ConfluenceResult,
    TimeframeView,
    format_confluence_report,
    multi_timeframe_vote,
)
from .fetchers import (
    BarFetcher,
    CSVBarFetcher,
    SubprocessBarFetcher,
    TradingViewMCPFetcher,
    make_fetcher,
)
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
from .watch import ChangeThresholds, MaterialChange, detect_material_change, watch_loop

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
    "BarFetcher",
    "CSVBarFetcher",
    "SubprocessBarFetcher",
    "TradingViewMCPFetcher",
    "make_fetcher",
    "ConfluenceResult",
    "TimeframeView",
    "multi_timeframe_vote",
    "format_confluence_report",
    "ChangeThresholds",
    "MaterialChange",
    "detect_material_change",
    "watch_loop",
    "fan_chart",
]


def __getattr__(name):
    if name == "PathForecaster":
        from .paths import PathForecaster

        return PathForecaster
    if name == "fan_chart":
        from .chart import fan_chart

        return fan_chart
    raise AttributeError(f"module 'signals' has no attribute {name!r}")
