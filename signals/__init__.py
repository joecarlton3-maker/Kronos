"""Probabilistic decision-support layer on top of Kronos forecasts.

`PathForecaster` is imported lazily so that `signals.stats` / `signals.report`
can be used without torch (e.g. when post-processing pre-computed paths).
"""
from .report import format_report
from .stats import PathStats, compute_stats

__all__ = ["PathForecaster", "PathStats", "compute_stats", "format_report"]


def __getattr__(name):
    if name == "PathForecaster":
        from .paths import PathForecaster

        return PathForecaster
    raise AttributeError(f"module 'signals' has no attribute {name!r}")
