"""Tests for fan-chart, confluence, watch loop, and fetchers."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from signals.confluence import format_confluence_report, multi_timeframe_vote
from signals.fetchers import (
    CSVBarFetcher,
    SubprocessBarFetcher,
    TradingViewMCPFetcher,
    make_fetcher,
)
from signals.stats import compute_stats
from signals.watch import (
    ChangeThresholds,
    detect_material_change,
    watch_loop,
)


def _hist(n: int = 100, last_close: float = 5000.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = last_close + np.cumsum(rng.normal(0, 1.0, n))
    closes -= closes[-1] - last_close
    highs = closes + 1.0
    lows = closes - 1.0
    opens = np.r_[closes[0], closes[:-1]]
    ts = pd.date_range("2024-01-01 09:30", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamps": ts, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": 1000.0,
    })


def _paths(n: int, T: int, drift: float, vol: float, start: float = 5000.0):
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


# --------------- fan chart ---------------

def test_fan_chart_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from signals.chart import fan_chart

    hist = _hist(120)
    paths = _paths(80, 30, drift=0.5, vol=1.0)
    stats = compute_stats(paths, hist, levels=[("VWAP", 5001.0)])
    out = tmp_path / "chart.png"
    fan_chart(stats, hist, out, symbol="ES", timeframe="5m",
              levels=[("VWAP", 5001.0)])
    assert out.exists() and out.stat().st_size > 1000


# --------------- confluence ---------------

def test_confluence_all_long_high_agreement():
    hist = _hist(100)
    bull_paths = _paths(100, 30, drift=1.0, vol=0.5)
    s = compute_stats(bull_paths, hist)
    res = multi_timeframe_vote([("1m", s), ("5m", s), ("15m", s)])
    assert res.aggregated_bias == "long"
    assert res.agreement_score == 1.0
    assert not res.conflict_flag


def test_confluence_detects_conflict():
    hist = _hist(100)
    bull = compute_stats(_paths(100, 30, drift=2.0, vol=0.5), hist)
    bear = compute_stats(_paths(100, 30, drift=-2.0, vol=0.5), hist)
    res = multi_timeframe_vote([("1m", bull), ("5m", bear)])
    assert res.conflict_flag is True
    assert any("CONFLICT" in n for n in res.notes)


def test_confluence_report_renders():
    hist = _hist(100)
    s = compute_stats(_paths(100, 30, drift=0.6, vol=0.6), hist)
    res = multi_timeframe_vote([("1m", s), ("5m", s)])
    text = format_confluence_report(res)
    assert "MULTI-TIMEFRAME CONFLUENCE" in text
    assert "Aggregated bias" in text


def test_confluence_empty_raises():
    with pytest.raises(ValueError):
        multi_timeframe_vote([])


# --------------- watch loop ---------------

def test_detect_material_change_first_forecast_signals():
    hist = _hist(100)
    s = compute_stats(_paths(50, 20, drift=0.0, vol=1.0), hist)
    change = detect_material_change(None, s)
    assert change
    assert "first" in change.reasons[0]


def test_detect_material_change_p_up_shift():
    hist = _hist(100)
    s_low = compute_stats(_paths(100, 20, drift=-0.5, vol=0.5), hist)
    s_high = compute_stats(_paths(100, 20, drift=1.0, vol=0.5), hist)
    th = ChangeThresholds(p_up_delta=0.10)
    change = detect_material_change(s_low, s_high, th)
    assert change
    assert any("P(up) shifted" in r or "bias flipped" in r for r in change.reasons)


def test_detect_material_change_no_change_when_stats_same():
    hist = _hist(100)
    s = compute_stats(_paths(100, 20, drift=0.0, vol=1.0), hist)
    change = detect_material_change(s, s)
    assert not change


def test_watch_loop_runs_capped_iterations():
    """Drive watch_loop with an in-memory forecaster and a fake sleep."""
    hist = _hist(100)
    base = compute_stats(_paths(100, 20, drift=0.0, vol=1.0), hist)
    forecasts = [
        base,
        replace(base, p_up_terminal=base.p_up_terminal + 0.30),
        base,
    ]
    sleep_calls = []
    alerts = []

    def forecast_fn():
        return forecasts.pop(0)

    def alert_fn(change, stats):
        alerts.append(change.reasons)

    watch_loop(
        forecast_fn,
        interval_seconds=0,
        alert_fn=alert_fn,
        thresholds=ChangeThresholds(p_up_delta=0.10),
        max_iterations=3,
        sleep_fn=sleep_calls.append,
    )
    # First iteration always alerts ("first forecast"), second alerts on shift.
    assert len(alerts) >= 2


def test_watch_loop_recovers_from_forecast_exception(capsys):
    hist = _hist(100)
    s = compute_stats(_paths(50, 20, drift=0.0, vol=1.0), hist)
    seq = iter([RuntimeError("boom"), s])

    def forecast_fn():
        v = next(seq)
        if isinstance(v, Exception):
            raise v
        return v

    watch_loop(
        forecast_fn, interval_seconds=0,
        max_iterations=2, sleep_fn=lambda _: None,
    )
    err = capsys.readouterr().err
    assert "forecast error" in err


# --------------- fetchers ---------------

def test_csv_fetcher_returns_validated_dataframe(tmp_path):
    csv = tmp_path / "bars.csv"
    df = _hist(50)
    df.to_csv(csv, index=False)
    fetcher = CSVBarFetcher(csv)
    out = fetcher.fetch_bars("X", "5m", lookback=20)
    assert len(out) == 20
    assert list(out.columns)[:5] == ["timestamps", "open", "high", "low", "close"]


def test_csv_fetcher_raises_on_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("a,b\n1,2\n")
    fetcher = CSVBarFetcher(csv)
    with pytest.raises(ValueError):
        fetcher.fetch_bars("X", "5m", lookback=10)


def test_make_fetcher_dispatches():
    f = make_fetcher("csv:/tmp/dummy.csv")
    assert isinstance(f, CSVBarFetcher)
    f = make_fetcher("subprocess:echo hi")
    assert isinstance(f, SubprocessBarFetcher)
    f = make_fetcher("tradingview")
    assert isinstance(f, TradingViewMCPFetcher)
    f = make_fetcher("tradingview:my-cmd --x")
    assert isinstance(f, TradingViewMCPFetcher)
    assert "my-cmd" in f.command_template
    with pytest.raises(ValueError):
        make_fetcher("nope:anything")


def test_subprocess_fetcher_executes_command_and_parses(tmp_path):
    """Use `cat` as a stand-in MCP CLI: it 'returns' whatever CSV we feed it."""
    csv = tmp_path / "bars.csv"
    df = _hist(30)
    df.to_csv(csv, index=False)
    # Template: cat the file regardless of substitution placeholders.
    fetcher = SubprocessBarFetcher(f"cat {csv}")
    out = fetcher.fetch_bars("ES", "5m", lookback=10)
    assert len(out) == 10
    assert "close" in out.columns


def test_subprocess_fetcher_fails_loudly_on_error():
    fetcher = SubprocessBarFetcher("false")
    with pytest.raises(RuntimeError):
        fetcher.fetch_bars("X", "5m", lookback=10)
