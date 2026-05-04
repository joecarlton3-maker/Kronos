"""Tests for the extended calibration analyzer (drawdown + timing)."""
import numpy as np
import pandas as pd

from signals.backtest import run_backtest
from signals.calibration import analyze, format_calibration_report


def _csv(n=600, drift=0.0, vol=1.0):
    rng = np.random.default_rng(0)
    closes = 5000 + np.cumsum(rng.normal(drift, vol, n))
    highs = closes + rng.uniform(0.5, 2.0, n)
    lows = closes - rng.uniform(0.5, 2.0, n)
    opens = np.r_[closes[0], closes[:-1]]
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamps": ts, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": 1000.0, "amount": 1000.0 * closes,
    })


def _good_forecaster(hist, x_ts, y_ts, pred_len, samples):
    """Forecaster that roughly matches the synthetic data's behavior."""
    rng = np.random.default_rng(int(abs(hist["close"].iloc[-1])) % 1000)
    last = float(hist["close"].iloc[-1])
    paths = np.zeros((samples, pred_len, 6))
    for i in range(samples):
        steps = rng.normal(0.0, 1.0, pred_len)
        cl = last + np.cumsum(steps)
        paths[i, :, 0] = np.r_[last, cl[:-1]]
        paths[i, :, 1] = cl + 1.0
        paths[i, :, 2] = cl - 1.0
        paths[i, :, 3] = cl
        paths[i, :, 4] = 1000.0
        paths[i, :, 5] = 1000.0 * cl
    return paths


def test_drawdown_calibration_present_and_sensible(tmp_path):
    df = _csv(n=600)
    log = tmp_path / "fcst.jsonl"
    run_backtest(
        df, forecaster_fn=_good_forecaster,
        lookback=100, horizon=20, samples=40, step=20,
        log_path=log,
    )
    rep = analyze(log)
    assert rep.drawdown_long is not None
    assert rep.drawdown_short is not None
    assert rep.drawdown_long.predicted_median > 0
    # Exceedance rates should be in [0, 1]
    for v in [
        rep.drawdown_long.rate_exceeded_p50,
        rep.drawdown_long.rate_exceeded_p75,
        rep.drawdown_long.rate_exceeded_p90,
    ]:
        assert 0.0 <= v <= 1.0


def test_timing_calibration_for_suggested_levels(tmp_path):
    df = _csv(n=600)
    log = tmp_path / "fcst.jsonl"
    run_backtest(
        df, forecaster_fn=_good_forecaster,
        lookback=100, horizon=20, samples=40, step=20,
        log_path=log,
    )
    rep = analyze(log)
    assert rep.timing_calibs is not None
    labels = [tc.label for tc in rep.timing_calibs]
    assert "Suggested target (long)" in labels
    assert "Suggested stop (long)" in labels
    for tc in rep.timing_calibs:
        assert 0.0 <= tc.predicted_p_touch_mean <= 1.0
        assert 0.0 <= tc.actual_touch_rate <= 1.0


def test_level_timing_calibration(tmp_path):
    df = _csv(n=600)
    log = tmp_path / "fcst.jsonl"
    run_backtest(
        df, forecaster_fn=_good_forecaster,
        lookback=100, horizon=20, samples=40, step=20,
        log_path=log, levels=[("Above", 5005.0), ("Below", 4995.0)],
    )
    rep = analyze(log)
    labels = [tc.label for tc in rep.level_timing_calibs]
    assert "Above" in labels
    assert "Below" in labels


def test_extended_report_renders(tmp_path):
    df = _csv(n=600)
    log = tmp_path / "fcst.jsonl"
    run_backtest(
        df, forecaster_fn=_good_forecaster,
        lookback=100, horizon=20, samples=40, step=20,
        log_path=log, levels=[("VWAP", 5000.0)],
    )
    rep = analyze(log)
    text = format_calibration_report(rep)
    for section in [
        "DRAWDOWN CALIBRATION",
        "TIMING CALIBRATION (suggested stops/targets)",
        "TIMING CALIBRATION (key levels)",
    ]:
        assert section in text, f"missing section: {section}"


def test_old_log_records_without_new_fields_still_load(tmp_path):
    """Records written before drawdown/timing were added should still load."""
    log = tmp_path / "old.jsonl"
    log.write_text(
        '{"ts_generated":"2024-01-01T00:00:00","ts_history_end":"2024-01-01T00:00:00",'
        '"ts_horizon_end":"2024-01-01T00:30:00","symbol":"X","timeframe":"5m",'
        '"lookback":100,"horizon":20,"samples":10,'
        '"stats":{"p_up_terminal":0.6,"last_price":5000,"level_touches":[],'
        '"level_time_to_touch":[]},'
        '"per_path_terminal_close":[5005],"per_path_max_high":[5010],'
        '"per_path_min_low":[4995],'
        '"outcome":{"actual_terminal_close":5005,"actual_max_high":5010,'
        '"actual_min_low":4995,"direction_up":true,'
        '"hit_target_long":false,"hit_stop_long":false,'
        '"long_first_touch":"neither","realized_r_long":0.5,'
        '"hit_target_short":false,"hit_stop_short":false,'
        '"short_first_touch":"neither","realized_r_short":-0.5,'
        '"level_actual_touches":[]}}\n'
    )
    from signals.log import iter_records

    records = list(iter_records(log))
    assert len(records) == 1
    o = records[0].outcome
    # New fields should default
    assert o.realized_dd_long == 0.0
    assert o.bars_to_target_long is None
    assert o.level_actual_bars_to_touch == []
