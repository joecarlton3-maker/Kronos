"""Tests for the backtest harness, log persistence, and calibration analyzer.

These avoid loading Kronos by injecting a synthetic forecaster.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signals.backtest import run_backtest
from signals.calibration import analyze, format_calibration_report
from signals.fill_outcomes import main as fill_outcomes_main
from signals.log import iter_records


def _make_synth_csv(n_bars: int = 600, drift: float = 0.05, vol: float = 1.0,
                    start: float = 5000.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = start + np.cumsum(rng.normal(drift, vol, n_bars))
    highs = closes + rng.uniform(0.5, 2.0, n_bars)
    lows = closes - rng.uniform(0.5, 2.0, n_bars)
    opens = np.r_[closes[0], closes[:-1]]
    ts = pd.date_range("2024-01-01", periods=n_bars, freq="5min")
    return pd.DataFrame({
        "timestamps": ts, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": 1000.0, "amount": 1000.0 * closes,
    })


def _bullish_forecaster(hist_df, x_ts, y_ts, pred_len, samples):
    rng = np.random.default_rng(int(abs(hist_df["close"].iloc[-1])) % 1000)
    last = float(hist_df["close"].iloc[-1])
    paths = np.zeros((samples, pred_len, 6))
    for i in range(samples):
        steps = rng.normal(0.4, 0.7, pred_len)
        cl = last + np.cumsum(steps)
        paths[i, :, 0] = np.r_[last, cl[:-1]]
        paths[i, :, 1] = cl + 1.0
        paths[i, :, 2] = cl - 1.0
        paths[i, :, 3] = cl
        paths[i, :, 4] = 1000
        paths[i, :, 5] = 1000 * cl
    return paths


def _flat_forecaster(hist_df, x_ts, y_ts, pred_len, samples):
    rng = np.random.default_rng(int(abs(hist_df["close"].iloc[-1])) % 1000)
    last = float(hist_df["close"].iloc[-1])
    paths = np.zeros((samples, pred_len, 6))
    for i in range(samples):
        steps = rng.normal(0.0, 1.5, pred_len)
        cl = last + np.cumsum(steps)
        paths[i, :, 0] = np.r_[last, cl[:-1]]
        paths[i, :, 1] = cl + 1.0
        paths[i, :, 2] = cl - 1.0
        paths[i, :, 3] = cl
        paths[i, :, 4] = 1000
        paths[i, :, 5] = 1000 * cl
    return paths


def test_backtest_writes_complete_records(tmp_path):
    df = _make_synth_csv(n_bars=400)
    log = tmp_path / "fcst.jsonl"
    n = run_backtest(
        df, forecaster_fn=_bullish_forecaster,
        lookback=100, horizon=20, samples=30, step=20,
        symbol="TEST", timeframe="5m",
        log_path=log, levels=[5005.0, 4995.0],
    )
    assert n > 0
    records = list(iter_records(log))
    assert len(records) == n
    for r in records:
        assert r.outcome is not None
        assert r.horizon == 20
        assert r.samples == 30
        assert r.outcome.long_first_touch in ("target", "stop", "neither")
        assert r.outcome.short_first_touch in ("target", "stop", "neither")
        assert isinstance(r.outcome.direction_up, bool)
        assert len(r.outcome.level_actual_touches) == 2


def test_first_touch_logic_stop_before_target():
    """If a bar has both stop and target inside its range, count as stop."""
    from signals.backtest import _evaluate_outcome

    bars = pd.DataFrame({
        "open": [100.0], "high": [110.0], "low": [90.0], "close": [105.0],
    })
    out = _evaluate_outcome(
        actual_bars=bars, last_price=100.0,
        suggested_stop_long=95.0, suggested_target_long=108.0,
        suggested_stop_short=105.0, suggested_target_short=92.0,
        levels=[],
    )
    assert out.long_first_touch == "stop"
    assert out.short_first_touch == "stop"
    assert out.realized_r_long == -1.0
    assert out.realized_r_short == -1.0


def test_first_touch_target_when_no_stop_hit():
    from signals.backtest import _evaluate_outcome

    bars = pd.DataFrame({
        "open": [100.0, 101.0], "high": [102.0, 109.0],
        "low":  [99.5,   100.5], "close": [101.0, 108.0],
    })
    out = _evaluate_outcome(
        actual_bars=bars, last_price=100.0,
        suggested_stop_long=95.0, suggested_target_long=108.0,
        suggested_stop_short=110.0, suggested_target_short=98.0,
        levels=[],
    )
    assert out.long_first_touch == "target"
    assert out.realized_r_long == pytest.approx((108.0 - 100.0) / (100.0 - 95.0))


def test_calibration_report_renders_and_is_sane(tmp_path):
    """A bullish forecaster on a bullish market should show non-trivial signal."""
    df = _make_synth_csv(n_bars=600, drift=0.10)
    log = tmp_path / "fcst.jsonl"
    run_backtest(
        df, forecaster_fn=_bullish_forecaster,
        lookback=100, horizon=20, samples=30, step=20,
        log_path=log,
    )
    rep = analyze(log)
    assert rep.n_forecasts > 5
    assert 0.0 <= rep.brier_score <= 1.0
    assert rep.log_loss > 0
    text = format_calibration_report(rep)
    for section in [
        "KRONOS CALIBRATION REPORT",
        "DIRECTIONAL CALIBRATION",
        "HIT RATE BY CONFIDENCE THRESHOLD",
        "EXCURSION CALIBRATION",
        "TRADE-PLAN PERFORMANCE",
        "INTERPRETATION",
    ]:
        assert section in text


def test_calibration_catches_overconfident_forecaster(tmp_path):
    """An over-confident forecaster (always predicts strong up) on a market
    that only goes up some of the time should have a worse Brier score than
    a well-spread forecaster, even if the market has mild positive drift.
    This is the whole point of the calibration analyzer: a high P(up) means
    nothing if it's not realized."""
    df = _make_synth_csv(n_bars=600, drift=0.10)
    overconf_log = tmp_path / "overconf.jsonl"
    spread_log = tmp_path / "spread.jsonl"
    run_backtest(
        df, forecaster_fn=_bullish_forecaster,    # always P(up) ~ 99%
        lookback=100, horizon=20, samples=30, step=20,
        log_path=overconf_log,
    )
    run_backtest(
        df, forecaster_fn=_flat_forecaster,        # P(up) spread around 50%
        lookback=100, horizon=20, samples=30, step=20,
        log_path=spread_log,
    )
    overconf = analyze(overconf_log)
    spread = analyze(spread_log)
    assert overconf.brier_score > spread.brier_score
    # Interpretation should flag overconfidence
    text = format_calibration_report(overconf)
    assert "over-confident" in text.lower() or overconf.brier_score >= 0.25


def test_no_records_raises(tmp_path):
    log = tmp_path / "empty.jsonl"
    log.write_text("")
    with pytest.raises(ValueError):
        analyze(log)


def test_fill_outcomes_populates_pending_records(tmp_path, monkeypatch, capsys):
    """Simulate a live forecast (no outcome) then fill it in."""
    from datetime import datetime
    from signals.log import record_from_paths, write_record
    from signals.stats import compute_stats

    df = _make_synth_csv(n_bars=300)
    hist = df.iloc[:200].reset_index(drop=True)
    paths = _bullish_forecaster(hist[["open", "high", "low", "close", "volume", "amount"]],
                                hist["timestamps"], df.iloc[200:230]["timestamps"], 30, 20)
    stats = compute_stats(paths, hist)
    rec = record_from_paths(
        paths, stats,
        ts_generated=datetime.now(),
        ts_history_end=hist["timestamps"].iloc[-1].to_pydatetime(),
        ts_horizon_end=df["timestamps"].iloc[229].to_pydatetime(),
        symbol="TEST", timeframe="5m", lookback=200,
    )
    log = tmp_path / "live.jsonl"
    write_record(log, rec)

    csv = tmp_path / "bars.csv"
    df.to_csv(csv, index=False)

    fill_outcomes_main(["--log", str(log), "--csv", str(csv)])

    records = list(iter_records(log))
    assert len(records) == 1
    assert records[0].outcome is not None
