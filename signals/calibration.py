"""Calibration analysis on a forecast log.

Answers: do Kronos's probability outputs reflect realized outcomes? Without
this, every probability in the forecast report is just a vibe.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .log import iter_records


LONG_BIAS_THRESHOLD = 0.55
SHORT_BIAS_THRESHOLD = 0.45


@dataclass
class CalibBucket:
    lo: float
    hi: float
    n: int
    predicted_mean: float
    actual_rate: float


@dataclass
class DrawdownCalib:
    side: str
    predicted_median: float
    predicted_p75: float
    predicted_p90: float
    realized_mean: float
    rate_exceeded_p50: float    # target ~50%
    rate_exceeded_p75: float    # target ~25%
    rate_exceeded_p90: float    # target ~10%


@dataclass
class TimingCalib:
    label: str
    n_records: int
    predicted_p_touch_mean: float
    actual_touch_rate: float
    n_actually_touched: int
    median_bars_predicted: float    # mean of predicted median across forecasts
    median_bars_actual: float       # NaN if no touches
    bars_bias: float                # actual - predicted; positive = touches arrive later than predicted


@dataclass
class CalibrationReport:
    n_forecasts: int
    horizon: int
    samples: int
    symbol: str
    timeframe: str
    date_start: str
    date_end: str

    brier_score: float
    log_loss: float
    buckets: List[CalibBucket]

    threshold_hit: List[dict]

    p75_high_realized: float
    p25_low_realized: float
    p90_high_realized: float
    p10_low_realized: float

    long_n: int
    long_win_rate: float
    long_avg_r: float
    short_n: int
    short_win_rate: float
    short_avg_r: float

    level_calib: List[dict]

    drawdown_long: Optional[DrawdownCalib] = None
    drawdown_short: Optional[DrawdownCalib] = None
    timing_calibs: List[TimingCalib] = None
    level_timing_calibs: List[TimingCalib] = None


def analyze(log_path: Path) -> CalibrationReport:
    records = [r for r in iter_records(log_path) if r.outcome is not None]
    if not records:
        raise ValueError(f"No records with outcomes found in {log_path}")

    p_up = np.array([r.stats["p_up_terminal"] for r in records])
    actual = np.array([1.0 if r.outcome.direction_up else 0.0 for r in records])

    brier = float(np.mean((p_up - actual) ** 2))
    p_clip = np.clip(p_up, 1e-6, 1 - 1e-6)
    log_loss = float(
        -np.mean(actual * np.log(p_clip) + (1 - actual) * np.log(1 - p_clip))
    )

    edges = np.linspace(0.0, 1.0, 11)
    buckets: List[CalibBucket] = []
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        if i < 9:
            mask = (p_up >= lo) & (p_up < hi)
        else:
            mask = (p_up >= lo) & (p_up <= hi)
        n = int(mask.sum())
        if n > 0:
            pmean = float(p_up[mask].mean())
            arate = float(actual[mask].mean())
        else:
            pmean = arate = float("nan")
        buckets.append(
            CalibBucket(
                lo=float(lo), hi=float(hi), n=n,
                predicted_mean=pmean, actual_rate=arate,
            )
        )

    threshold_hit = []
    for thr in [0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = p_up >= thr
        n = int(mask.sum())
        wr = float(actual[mask].mean()) if n > 0 else float("nan")
        threshold_hit.append({"threshold": thr, "n": n, "win_rate": wr})

    actual_high = np.array([r.outcome.actual_max_high for r in records])
    actual_low = np.array([r.outcome.actual_min_low for r in records])
    p75_pred_high = np.array(
        [np.percentile(r.per_path_max_high, 75) for r in records]
    )
    p90_pred_high = np.array(
        [np.percentile(r.per_path_max_high, 90) for r in records]
    )
    p25_pred_low = np.array(
        [np.percentile(r.per_path_min_low, 25) for r in records]
    )
    p10_pred_low = np.array(
        [np.percentile(r.per_path_min_low, 10) for r in records]
    )

    p75_high_realized = float((actual_high > p75_pred_high).mean())
    p90_high_realized = float((actual_high > p90_pred_high).mean())
    p25_low_realized = float((actual_low < p25_pred_low).mean())
    p10_low_realized = float((actual_low < p10_pred_low).mean())

    long_mask = p_up >= LONG_BIAS_THRESHOLD
    short_mask = p_up <= SHORT_BIAS_THRESHOLD
    all_long_r = np.array([r.outcome.realized_r_long for r in records])
    all_short_r = np.array([r.outcome.realized_r_short for r in records])
    long_r = all_long_r[long_mask]
    short_r = all_short_r[short_mask]

    long_n = int(long_mask.sum())
    short_n = int(short_mask.sum())
    long_win_rate = float((long_r > 0).mean()) if long_n else float("nan")
    short_win_rate = float((short_r > 0).mean()) if short_n else float("nan")
    long_avg_r = float(long_r.mean()) if long_n else float("nan")
    short_avg_r = float(short_r.mean()) if short_n else float("nan")

    level_calib = _level_calibration(records)

    sorted_by_ts = sorted(records, key=lambda r: r.ts_history_end)
    date_start = sorted_by_ts[0].ts_history_end
    date_end = sorted_by_ts[-1].ts_history_end

    drawdown_long_calib = _drawdown_calib(records, side="long")
    drawdown_short_calib = _drawdown_calib(records, side="short")
    timing_calibs = _suggested_timing_calibs(records)
    level_timing_calibs = _level_timing_calibs(records)

    return CalibrationReport(
        n_forecasts=len(records),
        horizon=records[0].horizon,
        samples=records[0].samples,
        symbol=records[0].symbol,
        timeframe=records[0].timeframe,
        date_start=date_start,
        date_end=date_end,
        brier_score=brier,
        log_loss=log_loss,
        buckets=buckets,
        threshold_hit=threshold_hit,
        p75_high_realized=p75_high_realized,
        p25_low_realized=p25_low_realized,
        p90_high_realized=p90_high_realized,
        p10_low_realized=p10_low_realized,
        long_n=long_n,
        long_win_rate=long_win_rate,
        long_avg_r=long_avg_r,
        short_n=short_n,
        short_win_rate=short_win_rate,
        short_avg_r=short_avg_r,
        level_calib=level_calib,
        drawdown_long=drawdown_long_calib,
        drawdown_short=drawdown_short_calib,
        timing_calibs=timing_calibs,
        level_timing_calibs=level_timing_calibs,
    )


def _drawdown_calib(records, side: str) -> Optional[DrawdownCalib]:
    """Compare predicted drawdown distribution to realized drawdown."""
    pred_med, pred_p75, pred_p90, realized = [], [], [], []
    for r in records:
        dd_key = f"drawdown_{side}"
        dd = r.stats.get(dd_key)
        if dd is None:
            continue
        pred_med.append(dd["median"])
        pred_p75.append(dd["p75"])
        pred_p90.append(dd["p90"])
        actual = (
            r.outcome.realized_dd_long if side == "long"
            else r.outcome.realized_dd_short
        )
        realized.append(actual)
    if not pred_med:
        return None
    pred_med = np.array(pred_med)
    pred_p75 = np.array(pred_p75)
    pred_p90 = np.array(pred_p90)
    realized = np.array(realized)
    return DrawdownCalib(
        side=side,
        predicted_median=float(pred_med.mean()),
        predicted_p75=float(pred_p75.mean()),
        predicted_p90=float(pred_p90.mean()),
        realized_mean=float(realized.mean()),
        rate_exceeded_p50=float((realized > pred_med).mean()),
        rate_exceeded_p75=float((realized > pred_p75).mean()),
        rate_exceeded_p90=float((realized > pred_p90).mean()),
    )


def _suggested_timing_calibs(records) -> List[TimingCalib]:
    """Calibration of predicted P_touch and median timing for the four
    suggested levels (long/short stop/target)."""
    out = []
    for stat_key, outcome_attr, pretty in [
        ("time_to_target_long", "bars_to_target_long", "Suggested target (long)"),
        ("time_to_stop_long", "bars_to_stop_long", "Suggested stop (long)"),
        ("time_to_target_short", "bars_to_target_short", "Suggested target (short)"),
        ("time_to_stop_short", "bars_to_stop_short", "Suggested stop (short)"),
    ]:
        calib = _timing_calib_for(records, stat_key, outcome_attr, pretty)
        if calib is not None:
            out.append(calib)
    return out


def _level_timing_calibs(records) -> List[TimingCalib]:
    """Calibration for each named key level."""
    by_label: dict = {}
    for r in records:
        ttts = r.stats.get("level_time_to_touch", [])
        bars_list = getattr(r.outcome, "level_actual_bars_to_touch", []) or []
        for ttt, actual_bar in zip(ttts, bars_list):
            label = ttt.get("label") or f"@{ttt.get('price', 0):.2f}"
            entry = by_label.setdefault(label, {
                "predicted_p": [], "predicted_med": [],
                "actual_touched": [], "actual_bars": [],
            })
            entry["predicted_p"].append(ttt.get("p_touched", 0.0))
            med = ttt.get("median_bars")
            if med is not None:
                entry["predicted_med"].append(med)
            entry["actual_touched"].append(1 if actual_bar is not None else 0)
            if actual_bar is not None:
                entry["actual_bars"].append(actual_bar)
    out = []
    for label, e in by_label.items():
        if not e["predicted_p"]:
            continue
        out.append(_finalize_timing(label, e))
    return out


def _timing_calib_for(records, stat_key, outcome_attr, label) -> Optional[TimingCalib]:
    e = {"predicted_p": [], "predicted_med": [], "actual_touched": [], "actual_bars": []}
    for r in records:
        ttt = r.stats.get(stat_key)
        if ttt is None:
            continue
        e["predicted_p"].append(ttt.get("p_touched", 0.0))
        med = ttt.get("median_bars")
        if med is not None:
            e["predicted_med"].append(med)
        actual = getattr(r.outcome, outcome_attr, None)
        e["actual_touched"].append(1 if actual is not None else 0)
        if actual is not None:
            e["actual_bars"].append(actual)
    if not e["predicted_p"]:
        return None
    return _finalize_timing(label, e)


def _finalize_timing(label: str, e: dict) -> TimingCalib:
    n_records = len(e["predicted_p"])
    n_touched = int(sum(e["actual_touched"]))
    pred_med_mean = float(np.mean(e["predicted_med"])) if e["predicted_med"] else float("nan")
    actual_med = (
        float(np.median(e["actual_bars"])) if e["actual_bars"] else float("nan")
    )
    if e["predicted_med"] and e["actual_bars"]:
        bias = actual_med - pred_med_mean
    else:
        bias = float("nan")
    return TimingCalib(
        label=label,
        n_records=n_records,
        predicted_p_touch_mean=float(np.mean(e["predicted_p"])),
        actual_touch_rate=n_touched / n_records if n_records else float("nan"),
        n_actually_touched=n_touched,
        median_bars_predicted=pred_med_mean,
        median_bars_actual=actual_med,
        bars_bias=bias,
    )


def _level_calibration(records) -> list:
    pairs = []
    for r in records:
        levels = r.stats.get("level_touches", [])
        actuals = r.outcome.level_actual_touches
        for lvl, a in zip(levels, actuals):
            pairs.append((lvl["prob"], int(a)))
    if not pairs:
        return []
    preds = np.array([p for p, _ in pairs])
    acts = np.array([a for _, a in pairs])
    edges = np.linspace(0.0, 1.0, 6)
    out = []
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        if i < 4:
            m = (preds >= lo) & (preds < hi)
        else:
            m = (preds >= lo) & (preds <= hi)
        n = int(m.sum())
        if n > 0:
            out.append({
                "lo": float(lo), "hi": float(hi), "n": n,
                "predicted_mean": float(preds[m].mean()),
                "actual_rate": float(acts[m].mean()),
            })
    return out


def format_calibration_report(rep: CalibrationReport) -> str:
    sep = "=" * 64
    sub = "-" * 64
    L = [
        sep,
        "  KRONOS CALIBRATION REPORT",
        sep,
        f"  Symbol:        {rep.symbol}",
        f"  Timeframe:     {rep.timeframe}",
        f"  Forecasts:     {rep.n_forecasts}",
        f"  Horizon:       {rep.horizon} bars",
        f"  Samples/fcst:  {rep.samples}",
        f"  Date range:    {rep.date_start[:19]} -> {rep.date_end[:19]}",
        "",
        "DIRECTIONAL CALIBRATION (P(up @ horizon end))",
        sub,
        f"  Brier score:   {rep.brier_score:.4f}   (lower=better; 0.25=random)",
        f"  Log loss:      {rep.log_loss:.4f}   (lower=better; 0.693=random)",
        "",
        "  Bucket           n     predicted   realized   diff",
    ]
    for b in rep.buckets:
        if b.n == 0:
            continue
        diff = b.actual_rate - b.predicted_mean
        L.append(
            f"  [{b.lo:.1f},{b.hi:.1f})    {b.n:>4}       {b.predicted_mean:.2f}       {b.actual_rate:.2f}    {diff:+.2f}"
        )
    L += ["", "HIT RATE BY CONFIDENCE THRESHOLD", sub]
    for h in rep.threshold_hit:
        wr = h["win_rate"]
        wr_s = f"{wr:.2%}" if not (isinstance(wr, float) and math.isnan(wr)) else "n/a"
        L.append(
            f"  P(up) >= {h['threshold']:.2f}    n={h['n']:>4}    win rate = {wr_s}"
        )
    L += [
        "",
        "EXCURSION CALIBRATION (target % in parens)",
        sub,
        f"  Actual high exceeded predicted P75:  {rep.p75_high_realized:.2%}  (target 25%)",
        f"  Actual high exceeded predicted P90:  {rep.p90_high_realized:.2%}  (target 10%)",
        f"  Actual low broke predicted P25:      {rep.p25_low_realized:.2%}  (target 25%)",
        f"  Actual low broke predicted P10:      {rep.p10_low_realized:.2%}  (target 10%)",
        "",
        f"TRADE-PLAN PERFORMANCE (filtered by bias: long@P>={LONG_BIAS_THRESHOLD:.2f}, short@P<={SHORT_BIAS_THRESHOLD:.2f})",
        sub,
        f"  LONG  : n={rep.long_n:>4}   win rate={_pct(rep.long_win_rate)}   avg R={_fnum(rep.long_avg_r, sign=True)}",
        f"  SHORT : n={rep.short_n:>4}   win rate={_pct(rep.short_win_rate)}   avg R={_fnum(rep.short_avg_r, sign=True)}",
        "",
    ]
    if rep.level_calib:
        L += ["LEVEL-TOUCH CALIBRATION", sub]
        for b in rep.level_calib:
            L.append(
                f"  P_touch in [{b['lo']:.1f},{b['hi']:.1f}):  n={b['n']:>4}  predicted={b['predicted_mean']:.2f}  actual={b['actual_rate']:.2f}"
            )
        L.append("")
    if rep.drawdown_long is not None or rep.drawdown_short is not None:
        L += ["DRAWDOWN CALIBRATION (target rate in parens)", sub]
        for dd in (rep.drawdown_long, rep.drawdown_short):
            if dd is None:
                continue
            L.append(f"  {dd.side.upper()} entry @ last price:")
            L.append(
                f"    Predicted heat:  median={dd.predicted_median:.2f}  "
                f"P75={dd.predicted_p75:.2f}  P90={dd.predicted_p90:.2f}"
            )
            L.append(f"    Realized mean:   {dd.realized_mean:.2f}")
            L.append(
                f"    Realized exceeded predicted median:  {dd.rate_exceeded_p50:.2%}  (target 50%)"
            )
            L.append(
                f"    Realized exceeded predicted P75:     {dd.rate_exceeded_p75:.2%}  (target 25%)"
            )
            L.append(
                f"    Realized exceeded predicted P90:     {dd.rate_exceeded_p90:.2%}  (target 10%)"
            )
        L.append("")
    if rep.timing_calibs:
        L += ["TIMING CALIBRATION (suggested stops/targets)", sub]
        for tc in rep.timing_calibs:
            L += _format_timing(tc)
        L.append("")
    if rep.level_timing_calibs:
        L += ["TIMING CALIBRATION (key levels)", sub]
        for tc in rep.level_timing_calibs:
            L += _format_timing(tc)
        L.append("")
    L += ["INTERPRETATION", sub] + _interpretation(rep) + [sep]
    return "\n".join(L)


def _format_timing(tc: TimingCalib) -> List[str]:
    rate_diff = tc.actual_touch_rate - tc.predicted_p_touch_mean
    return [
        f"  {tc.label}",
        f"    n={tc.n_records}  predicted P_touch={tc.predicted_p_touch_mean:.2%}  "
        f"actual={tc.actual_touch_rate:.2%}  ({rate_diff:+.2%})",
        f"    predicted median bars={_fnum(tc.median_bars_predicted)}  "
        f"actual={_fnum(tc.median_bars_actual)}  bias={_fnum(tc.bars_bias, sign=True)}",
    ]


def _pct(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.2%}"


def _fnum(v: float, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.2f}" if sign else f"{v:.2f}"


def _interpretation(rep: CalibrationReport) -> List[str]:
    out = []
    if rep.brier_score < 0.20:
        out.append("  - Brier < 0.20: probabilities have signal vs random.")
    elif rep.brier_score < 0.25:
        out.append("  - Brier 0.20-0.25: marginal signal; tread carefully.")
    else:
        out.append("  - Brier >= 0.25: probabilities at-or-worse than random; do NOT trade on them.")

    diffs = [
        b.actual_rate - b.predicted_mean
        for b in rep.buckets
        if b.n > 0 and not math.isnan(b.actual_rate)
    ]
    if diffs:
        mean_diff = sum(diffs) / len(diffs)
        if mean_diff > 0.05:
            out.append("  - Model under-confident overall (actual > predicted).")
        elif mean_diff < -0.05:
            out.append("  - Model over-confident overall (actual < predicted). Tighten thresholds.")

    if not math.isnan(rep.long_avg_r) and rep.long_avg_r > 0.10:
        out.append(f"  - Long-side suggestion has positive avg R ({rep.long_avg_r:+.2f}). Watch for cost drag.")
    if not math.isnan(rep.short_avg_r) and rep.short_avg_r > 0.10:
        out.append(f"  - Short-side suggestion has positive avg R ({rep.short_avg_r:+.2f}).")

    if rep.p75_high_realized > 0.35:
        out.append("  - Upside excursion under-forecast: P75 exceeded too often. Targets too tight.")
    if rep.p10_low_realized > 0.18:
        out.append("  - Downside tail risk under-forecast: P10 stops hit too often.")
    for dd in (rep.drawdown_long, rep.drawdown_short):
        if dd is None:
            continue
        if dd.rate_exceeded_p90 > 0.20:
            out.append(
                f"  - {dd.side.upper()} drawdown tails under-forecast "
                f"(P90 exceeded {dd.rate_exceeded_p90:.0%} vs 10% target). Heat is worse than model predicts."
            )
    for tc in rep.timing_calibs or []:
        if not math.isnan(tc.bars_bias) and tc.bars_bias > 5:
            out.append(
                f"  - {tc.label}: touches arrive ~{tc.bars_bias:.0f} bars later than predicted on avg."
            )

    if rep.n_forecasts < 50:
        out.append(f"  - Sample size {rep.n_forecasts} is small. Treat all numbers as preliminary.")
    out.append("  - Re-run after fine-tuning Kronos on your contract.")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Calibration analysis of a Kronos forecast log"
    )
    p.add_argument("--log", required=True)
    args = p.parse_args(argv)
    rep = analyze(Path(args.log))
    print(format_calibration_report(rep))


if __name__ == "__main__":
    main()
