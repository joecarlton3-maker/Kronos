"""Tests for trajectory bands, drawdown, time-to-touch, plan evaluation,
auto key levels, and the report's new sections / JSON serialization."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from signals.cli import _json_default, _parse_plan
from signals.levels import auto_key_levels
from signals.plan import evaluate_plan
from signals.report import format_report
from signals.stats import compute_stats
from signals.trajectory import (
    drawdown_long,
    drawdown_short,
    per_bar_band,
    time_to_touch,
)


def _hist(n: int = 100, last_close: float = 5000.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = last_close + np.cumsum(rng.normal(0, 1.5, n))
    closes -= closes[-1] - last_close
    highs = closes + 1.0
    lows = closes - 1.0
    opens = np.r_[closes[0], closes[:-1]]
    ts = pd.date_range("2024-01-01 09:30", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamps": ts, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": np.full(n, 1000.0),
    })


def _paths(n: int, T: int, drift: float, vol: float, start: float = 5000.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    paths = np.zeros((n, T, 6))
    for i in range(n):
        steps = rng.normal(drift, vol, T)
        cl = start + np.cumsum(steps)
        paths[i, :, 0] = np.r_[start, cl[:-1]]
        paths[i, :, 1] = cl + vol
        paths[i, :, 2] = cl - vol
        paths[i, :, 3] = cl
        paths[i, :, 4] = 1000.0
        paths[i, :, 5] = 1000.0 * cl
    return paths


# --------------- trajectory ---------------

def test_per_bar_band_dimensions_and_ordering():
    paths = _paths(200, 30, drift=0.5, vol=1.0)
    band = per_bar_band(paths[:, :, 3])
    assert len(band.p10) == len(band.p50) == len(band.p90) == 30
    for i in range(30):
        assert band.p10[i] <= band.p25[i] <= band.p50[i] <= band.p75[i] <= band.p90[i]
    # Drift should make later bars trend higher
    assert band.p50[-1] > band.p50[0]


def test_drawdown_distribution():
    paths = _paths(500, 30, drift=0.0, vol=2.0)
    dd_long = drawdown_long(paths, 5000.0)
    dd_short = drawdown_short(paths, 5000.0)
    assert dd_long.median > 0
    assert dd_long.p25 <= dd_long.p75 <= dd_long.p90 <= dd_long.max_observed
    # Symmetric drift -> roughly comparable long/short drawdown
    assert abs(dd_long.median - dd_short.median) < dd_long.median


def test_time_to_touch_above_below():
    paths = _paths(300, 30, drift=0.0, vol=1.5)
    above = time_to_touch(paths, 5010.0, 5000.0, "Above")
    below = time_to_touch(paths, 4990.0, 5000.0, "Below")
    assert above.direction == "above"
    assert below.direction == "below"
    assert 0.0 <= above.p_touched <= 1.0
    if above.median_bars is not None:
        assert 1 <= above.median_bars <= 30
    # First-quarter touch <= first-half touch <= overall touch
    assert above.p_touched_first_quarter <= above.p_touched_first_half <= above.p_touched


def test_time_to_touch_unreachable_level():
    paths = _paths(100, 20, drift=0.0, vol=0.5)
    # 50 points away with vol 0.5 — extremely unlikely
    far = time_to_touch(paths, 5050.0, 5000.0, "Far")
    assert far.p_touched < 0.05
    assert far.median_bars is None or far.median_bars >= 1


# --------------- plan evaluation ---------------

def test_plan_long_favorable():
    """Strong upward drift, target close, stop far -> high P(target)."""
    paths = _paths(500, 30, drift=1.0, vol=0.8)
    plan = evaluate_plan(paths, entry=5000.0, stop=4990.0, target=5010.0, side="long")
    assert plan.p_target_first > plan.p_stop_first
    assert plan.expected_r > 0
    assert plan.median_bars_to_target is not None
    assert plan.median_bars_to_target <= 30


def test_plan_short_with_downward_paths():
    paths = _paths(500, 30, drift=-1.0, vol=0.8)
    plan = evaluate_plan(paths, entry=5000.0, stop=5010.0, target=4990.0, side="short")
    assert plan.p_target_first > plan.p_stop_first
    assert plan.expected_r > 0


def test_plan_invalid_levels_raises():
    paths = _paths(50, 20, drift=0.0, vol=1.0)
    with pytest.raises(ValueError):
        evaluate_plan(paths, entry=5000, stop=5010, target=5020, side="long")
    with pytest.raises(ValueError):
        evaluate_plan(paths, entry=5000, stop=4990, target=4980, side="short")
    with pytest.raises(ValueError):
        evaluate_plan(paths, entry=5000, stop=4990, target=5010, side="diagonal")


def test_plan_probabilities_sum_to_one():
    paths = _paths(200, 20, drift=0.2, vol=1.0)
    plan = evaluate_plan(paths, entry=5000, stop=4995, target=5005, side="long")
    total = plan.p_target_first + plan.p_stop_first + plan.p_neither
    assert abs(total - 1.0) < 1e-9


# --------------- auto key levels ---------------

def test_auto_key_levels_finds_session_and_prior_day():
    n = 200
    rng = np.random.default_rng(0)
    closes = 5000 + np.cumsum(rng.normal(0, 1.0, n))
    highs = closes + 1.0
    lows = closes - 1.0
    opens = np.r_[closes[0], closes[:-1]]
    ts = pd.date_range("2024-01-01 09:30", periods=n, freq="5min")
    df = pd.DataFrame({
        "timestamps": ts, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": 1000.0,
    })
    levels = auto_key_levels(df)
    labels = [l for l, _ in levels]
    assert "Session VWAP" in labels
    assert "Today High" in labels
    assert "Today Low" in labels
    assert "Prior Day High" in labels
    assert "Prior Close" in labels


def test_auto_key_levels_handles_missing_volume():
    ts = pd.date_range("2024-01-01 09:30", periods=20, freq="5min")
    df = pd.DataFrame({
        "timestamps": ts,
        "open": np.linspace(5000, 5005, 20),
        "high": np.linspace(5001, 5006, 20),
        "low": np.linspace(4999, 5004, 20),
        "close": np.linspace(5000, 5005, 20),
    })
    levels = auto_key_levels(df)
    labels = [l for l, _ in levels]
    assert "Session VWAP" not in labels  # no volume column
    assert "Today High" in labels


# --------------- report integration ---------------

def test_report_includes_new_sections():
    hist = _hist(120)
    paths = _paths(100, 30, drift=0.5, vol=1.0)
    stats = compute_stats(paths, hist, levels=[("Test Lvl", 5005.0)])
    report = format_report(stats, symbol="ES", timeframe="5m", lookback_bars=120)
    for section in ["TRAJECTORY", "DRAWDOWN PROFILE", "Time-to-target", "Time-to-stop"]:
        assert section in report
    assert "Test Lvl" in report
    assert "first-quarter=" in report  # time-to-touch line for level


def test_report_with_plan_section():
    hist = _hist(120)
    paths = _paths(200, 30, drift=0.5, vol=1.0)
    stats = compute_stats(paths, hist)
    plan = evaluate_plan(paths, entry=5000.0, stop=4995.0, target=5010.0, side="long")
    report = format_report(stats, symbol="ES", timeframe="5m",
                           lookback_bars=120, plan=plan)
    assert "YOUR PLAN (LONG)" in report
    assert "Expected R" in report
    assert "Verdict" in report


# --------------- CLI helpers ---------------

def test_parse_plan_round_trip():
    parsed = _parse_plan("side=long,entry=5142,stop=5128,target=5180")
    assert parsed == {"side": "long", "entry": 5142.0, "stop": 5128.0, "target": 5180.0}


def test_parse_plan_missing_key_errors():
    with pytest.raises(SystemExit):
        _parse_plan("side=long,entry=5000")


def test_json_default_handles_numpy_and_nan():
    assert _json_default(np.float32(1.5)) == 1.5
    assert _json_default(np.int64(7)) == 7
    arr = np.array([1.0, 2.0])
    assert _json_default(arr) == [1.0, 2.0]
    # NaN -> None for round-trippable JSON
    assert _json_default(np.float64("nan")) is None


def test_json_serialization_of_full_stats_succeeds():
    hist = _hist(120)
    paths = _paths(50, 20, drift=0.3, vol=1.0)
    stats = compute_stats(paths, hist, levels=[("VWAP", 5002.0)])
    plan = evaluate_plan(paths, entry=5000.0, stop=4995.0, target=5010.0, side="long")
    payload = {"stats": stats.__dict__, "plan": plan.__dict__}
    # asdict-style serialization via custom default
    text = json.dumps(payload, default=_json_default_recursive)
    parsed = json.loads(text)
    assert "stats" in parsed
    assert "plan" in parsed
    assert "expected_r" in parsed["plan"]


def _json_default_recursive(obj):
    """Recursive variant for nested dataclasses in tests."""
    if hasattr(obj, "__dataclass_fields__"):
        return obj.__dict__
    return _json_default(obj)
