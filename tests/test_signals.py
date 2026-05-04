"""Tests for the decision-support stats / report layer.

These do not load Kronos. They validate stats math and report formatting
against synthetic forecast paths.
"""
import numpy as np
import pandas as pd
import pytest

from signals.report import format_report
from signals.stats import compute_stats


def _synth_history(n: int = 100, last_close: float = 5000.0, atr: float = 5.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = last_close + np.cumsum(rng.normal(0, atr / 2, size=n))
    closes = closes - (closes[-1] - last_close)  # pin last close
    highs = closes + atr / 2
    lows = closes - atr / 2
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})


def _synth_paths(n_samples: int, pred_len: int, drift: float, vol: float,
                 start: float = 5000.0) -> np.ndarray:
    """OHLCVA paths with controllable drift and vol."""
    rng = np.random.default_rng(42)
    closes = np.zeros((n_samples, pred_len))
    for i in range(n_samples):
        steps = rng.normal(drift, vol, size=pred_len)
        closes[i] = start + np.cumsum(steps)
    highs = closes + vol
    lows = closes - vol
    opens = np.concatenate([np.full((n_samples, 1), start), closes[:, :-1]], axis=1)
    vols = np.full_like(closes, 1000.0)
    amts = vols * closes
    return np.stack([opens, highs, lows, closes, vols, amts], axis=-1)


def test_bullish_paths_produce_long_bias():
    hist = _synth_history()
    paths = _synth_paths(200, 60, drift=2.0, vol=1.0)
    stats = compute_stats(paths, hist)
    assert stats.p_up_terminal > 0.9
    assert stats.median_terminal > stats.last_price
    assert stats.pct_trend_up > stats.pct_trend_down


def test_bearish_paths_produce_short_bias():
    hist = _synth_history()
    paths = _synth_paths(200, 60, drift=-2.0, vol=1.0)
    stats = compute_stats(paths, hist)
    assert stats.p_up_terminal < 0.1
    assert stats.median_terminal < stats.last_price
    assert stats.pct_trend_down > stats.pct_trend_up


def test_neutral_high_vol_paths_produce_neutral_bias():
    hist = _synth_history()
    paths = _synth_paths(500, 60, drift=0.0, vol=5.0)
    stats = compute_stats(paths, hist)
    assert 0.4 < stats.p_up_terminal < 0.6
    assert stats.terminal_std > 10.0


def test_touch_probabilities_are_monotonic_in_distance():
    hist = _synth_history()
    paths = _synth_paths(500, 60, drift=0.0, vol=2.0, start=5000.0)
    stats = compute_stats(paths, hist, levels=[5005.0, 5050.0, 4995.0, 4950.0])
    above = [t for t in stats.level_touches if t.direction == "above"]
    below = [t for t in stats.level_touches if t.direction == "below"]
    # Closer = higher touch probability
    assert above[0].prob >= above[1].prob
    assert below[0].prob >= below[1].prob


def test_stats_shapes_and_quartiles():
    hist = _synth_history()
    paths = _synth_paths(100, 80, drift=0.5, vol=1.0)
    stats = compute_stats(paths, hist)
    assert stats.sample_count == 100
    assert stats.horizon_bars == 80
    assert len(stats.p_up_quartiles) == 4
    assert stats.terminal_p10 <= stats.terminal_p25 <= stats.terminal_p50 \
        <= stats.terminal_p75 <= stats.terminal_p90
    assert stats.iqr >= 0
    assert stats.atr_recent > 0


def test_report_renders_all_sections():
    hist = _synth_history()
    paths = _synth_paths(150, 60, drift=1.0, vol=1.5)
    stats = compute_stats(paths, hist, levels=[5050.0, 4950.0])
    report = format_report(stats, symbol="ES", timeframe="5m", lookback_bars=400)
    for section in [
        "KRONOS PROBABILISTIC FORECAST",
        "DIRECTIONAL BIAS",
        "EXPECTED RANGE",
        "EXCURSION ENVELOPE",
        "VOLATILITY & CONFIDENCE",
        "PATH SHAPE DISTRIBUTION",
        "KEY LEVEL TOUCH PROBABILITIES",
        "TRADE PLAN",
        "CAVEATS",
    ]:
        assert section in report, f"missing section: {section}"


def test_compute_stats_rejects_bad_shape():
    hist = _synth_history()
    with pytest.raises(ValueError):
        compute_stats(np.zeros((10, 5)), hist)
