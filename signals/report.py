"""Pretty-print PathStats as a terminal-friendly futures decision report."""
from __future__ import annotations

import math
from datetime import datetime
from typing import List, Optional

from .plan import PlanEvaluation
from .stats import PathStats
from .trajectory import TimeToTouch, TrajectoryBand


def format_report(
    stats: PathStats,
    symbol: str,
    timeframe: str,
    lookback_bars: int,
    generated_at: Optional[datetime] = None,
    plan: Optional[PlanEvaluation] = None,
) -> str:
    generated_at = generated_at or datetime.now()
    sep = "=" * 64
    sub = "-" * 64
    bias = _bias_label(stats.p_up_terminal)
    regime = _regime_label(stats)
    confidence = _confidence_label(stats.dispersion_in_atr)

    lines = [
        sep,
        "  KRONOS PROBABILISTIC FORECAST",
        sep,
        f"  Symbol:        {symbol}",
        f"  Timeframe:     {timeframe}",
        f"  Generated:     {generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Lookback:      {lookback_bars} bars",
        f"  Horizon:       {stats.horizon_bars} bars",
        f"  Samples:       {stats.sample_count}",
        f"  Last price:    {stats.last_price:.2f}",
        f"  ATR(14):       {stats.atr_recent:.2f}",
        "",
        "DIRECTIONAL BIAS",
        sub,
        f"  P(up @ horizon end):       {stats.p_up_terminal:>6.1%}",
    ]

    qlabels = ["25%", "50%", "75%", "100%"]
    for lbl, p in zip(qlabels, stats.p_up_quartiles):
        lines.append(f"  P(up @ {lbl:>4} of horizon):  {p:>6.1%}")
    lines += [
        f"  Median terminal close:     {stats.median_terminal:>10.2f}  ({_pct(stats.median_terminal_pct)})",
        f"  Mean terminal close:       {stats.mean_terminal:>10.2f}",
        f"  Bias:                      {bias}",
        "",
        "EXPECTED RANGE (terminal close percentiles)",
        sub,
    ]
    for label, val in [
        ("P10", stats.terminal_p10),
        ("P25", stats.terminal_p25),
        ("P50", stats.terminal_p50),
        ("P75", stats.terminal_p75),
        ("P90", stats.terminal_p90),
    ]:
        delta = val - stats.last_price
        lines.append(f"  {label}:  {val:>10.2f}   ({delta:+.2f})")
    lines += [
        f"  IQR (P75-P25):             {stats.iqr:.2f} pts",
        "",
        "TRAJECTORY (median close at horizon milestones)",
        sub,
    ]
    lines += _milestone_lines(stats.trajectory_close, stats.last_price, stats.horizon_bars)
    lines += [
        "",
        "EXCURSION ENVELOPE (across full horizon)",
        sub,
        f"  Median max high:           {stats.median_max_high:>10.2f}  ({stats.median_max_high - stats.last_price:+.2f})",
        f"  Median min low:            {stats.median_min_low:>10.2f}  ({stats.median_min_low - stats.last_price:+.2f})",
        f"  90% of paths stay below:   {stats.p90_max_high:>10.2f}",
        f"  90% of paths stay above:   {stats.p10_min_low:>10.2f}",
        "",
        "DRAWDOWN PROFILE (heat if held to horizon, in price points)",
        sub,
        "  LONG entry @ last price:",
        f"    Median heat:             {stats.drawdown_long.median:>8.2f}",
        f"    75% of paths see >=:     {stats.drawdown_long.p25:>8.2f}",
        f"    25% of paths see worse:  {stats.drawdown_long.p75:>8.2f}",
        f"    Tail (P90):              {stats.drawdown_long.p90:>8.2f}",
        "  SHORT entry @ last price:",
        f"    Median heat:             {stats.drawdown_short.median:>8.2f}",
        f"    75% of paths see >=:     {stats.drawdown_short.p25:>8.2f}",
        f"    25% of paths see worse:  {stats.drawdown_short.p75:>8.2f}",
        f"    Tail (P90):              {stats.drawdown_short.p90:>8.2f}",
        "",
        "VOLATILITY & CONFIDENCE",
        sub,
        f"  Terminal close std:        {stats.terminal_std:.2f}  ({stats.terminal_std_pct:.2f}% of price)",
        f"  Dispersion / ATR(14):      {_fnum(stats.dispersion_in_atr)}",
        f"  Confidence:                {confidence}",
        "",
        "PATH SHAPE DISTRIBUTION",
        sub,
        f"  Trend up:      {stats.pct_trend_up:>6.1%}",
        f"  Trend down:    {stats.pct_trend_down:>6.1%}",
        f"  Mean revert:   {stats.pct_mean_revert:>6.1%}",
        f"  Chop:          {stats.pct_chop:>6.1%}",
        f"  Dominant regime:           {regime}",
        "",
    ]

    if stats.level_touches:
        lines += ["KEY LEVEL TOUCH PROBABILITIES", sub]
        for t, ttt in zip(stats.level_touches, stats.level_time_to_touch):
            tag = _touch_label(t.prob)
            label = f"{t.label} " if t.label else ""
            lines.append(
                f"  {label}{t.level:.2f} ({t.direction}):  P_touch={t.prob:>5.1%}  -> {tag}"
            )
            lines.append(_ttt_line(ttt))
        lines.append("")

    lines += [
        "TRADE PLAN (suggestions only - not orders)",
        sub,
        "  LONG:",
        f"    Stop  (P15 path low):    {stats.suggested_stop_long:>10.2f}  (risk {stats.last_price - stats.suggested_stop_long:.2f})",
        f"    Target (P75 path high):  {stats.suggested_target_long:>10.2f}  (reward {stats.suggested_target_long - stats.last_price:.2f})",
        f"    Reward:Risk:             {_fnum(stats.rr_long)} : 1",
        f"    Time-to-target (median): {_fnum_bars(stats.time_to_target_long.median_bars)} bars   "
        f"(P_touch {_pct100(stats.time_to_target_long.p_touched)})",
        f"    Time-to-stop   (median): {_fnum_bars(stats.time_to_stop_long.median_bars)} bars   "
        f"(P_touch {_pct100(stats.time_to_stop_long.p_touched)})",
        "  SHORT:",
        f"    Stop  (P85 path high):   {stats.suggested_stop_short:>10.2f}  (risk {stats.suggested_stop_short - stats.last_price:.2f})",
        f"    Target (P25 path low):   {stats.suggested_target_short:>10.2f}  (reward {stats.last_price - stats.suggested_target_short:.2f})",
        f"    Reward:Risk:             {_fnum(stats.rr_short)} : 1",
        f"    Time-to-target (median): {_fnum_bars(stats.time_to_target_short.median_bars)} bars   "
        f"(P_touch {_pct100(stats.time_to_target_short.p_touched)})",
        f"    Time-to-stop   (median): {_fnum_bars(stats.time_to_stop_short.median_bars)} bars   "
        f"(P_touch {_pct100(stats.time_to_stop_short.p_touched)})",
        "",
    ]

    if plan is not None:
        lines += _plan_section(plan)

    lines += [
        "CAVEATS",
        sub,
        "  - Pretrained weights, not fine-tuned for your contract.",
        "  - Tail events under-represented; do not treat 0% as impossible.",
        "  - Re-run on regime shifts and avoid scheduled news windows.",
        "  - Decision support only - you make the trade.",
        sep,
    ]
    return "\n".join(lines)


def _milestone_lines(band: TrajectoryBand, last_price: float, horizon: int) -> List[str]:
    out = []
    quartiles = [0.25, 0.50, 0.75, 1.00]
    for q in quartiles:
        i = max(1, int(horizon * q)) - 1
        med = band.p50[i]
        lo = band.p10[i]
        hi = band.p90[i]
        out.append(
            f"  @ {int(q * 100):>3}% horizon (bar {i + 1:>3}):  median={med:.2f}  "
            f"({med - last_price:+.2f})  "
            f"P10..P90 = [{lo:.2f}, {hi:.2f}]"
        )
    return out


def _ttt_line(ttt: TimeToTouch) -> str:
    if ttt.median_bars is None:
        timing = "no median (low touch rate)"
    else:
        timing = f"median touch in {ttt.median_bars:.0f} bars"
    return (
        f"     -> P_touch={ttt.p_touched:>5.1%}, {timing}, "
        f"first-quarter={ttt.p_touched_first_quarter:.0%}, "
        f"first-half={ttt.p_touched_first_half:.0%}"
    )


def _plan_section(plan: PlanEvaluation) -> List[str]:
    sub = "-" * 64
    side_u = plan.side.upper()
    out = [
        f"YOUR PLAN ({side_u})",
        sub,
        f"  Entry={plan.entry:.2f}  Stop={plan.stop:.2f}  Target={plan.target:.2f}",
        f"  Risk per unit={plan.risk_per_unit:.2f}  Reward per unit={plan.reward_per_unit:.2f}  "
        f"Planned R:R = {plan.rr_planned:.2f} : 1",
        "",
        f"  P(target first):    {plan.p_target_first:>6.1%}",
        f"  P(stop first):      {plan.p_stop_first:>6.1%}",
        f"  P(neither, exit at horizon close): {plan.p_neither:>6.1%}",
        "",
        f"  Expected R:         {plan.expected_r:+.3f}",
        f"    avg R when target hit:   {_fnum(plan.avg_r_when_target, sign=True)}",
        f"    avg R when neither hit:  {_fnum(plan.avg_r_when_neither, sign=True)}",
        "",
        f"  Median bars to target:      {_fnum_bars(plan.median_bars_to_target)}",
        f"  Median bars to stop:        {_fnum_bars(plan.median_bars_to_stop)}",
        f"  Median bars to resolution:  {_fnum_bars(plan.median_bars_to_resolution)}",
        f"  Verdict:            {_plan_verdict(plan)}",
        "",
    ]
    return out


def _plan_verdict(plan: PlanEvaluation) -> str:
    if plan.expected_r >= 0.30:
        return f"FAVORABLE (E[R]={plan.expected_r:+.2f}) - aligns with model"
    if plan.expected_r >= 0.10:
        return f"SLIGHT EDGE (E[R]={plan.expected_r:+.2f}) - thin; account for costs"
    if plan.expected_r >= -0.10:
        return f"NEUTRAL (E[R]={plan.expected_r:+.2f}) - no model edge"
    return f"UNFAVORABLE (E[R]={plan.expected_r:+.2f}) - model disagrees"


def _pct(v: float) -> str:
    return f"{v:+.2f}%"


def _pct100(v: float) -> str:
    return f"{v:.0%}" if v == v else "n/a"


def _fnum(v: Optional[float], sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.2f}" if sign else f"{v:.2f}"


def _fnum_bars(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.0f}"


def _bias_label(p: float) -> str:
    if p >= 0.70:
        return f"LONG (high conviction, {p:.0%})"
    if p >= 0.60:
        return f"LONG (moderate, {p:.0%})"
    if p >= 0.55:
        return f"LONG (weak, {p:.0%})"
    if p <= 0.30:
        return f"SHORT (high conviction, {1 - p:.0%})"
    if p <= 0.40:
        return f"SHORT (moderate, {1 - p:.0%})"
    if p <= 0.45:
        return f"SHORT (weak, {1 - p:.0%})"
    return f"NEUTRAL ({p:.0%}) - stand aside"


def _confidence_label(disp_atr: float) -> str:
    if disp_atr is None or (isinstance(disp_atr, float) and math.isnan(disp_atr)):
        return "UNKNOWN"
    if disp_atr < 1.0:
        return "HIGH (tight path bundle)"
    if disp_atr < 2.0:
        return "NORMAL"
    if disp_atr < 3.0:
        return "LOW (wide dispersion)"
    return "VERY LOW (consider standing aside)"


def _regime_label(stats: PathStats) -> str:
    trending = stats.pct_trend_up + stats.pct_trend_down
    if trending >= 0.55:
        return f"TRENDING ({trending:.0%} directional)"
    if stats.pct_mean_revert >= 0.40:
        return f"MEAN-REVERTING ({stats.pct_mean_revert:.0%})"
    if stats.pct_chop >= 0.40:
        return f"CHOP ({stats.pct_chop:.0%}) - consider standing aside"
    return "MIXED"


def _touch_label(p: float) -> str:
    if p >= 0.70:
        return "very likely test"
    if p >= 0.50:
        return "likely test"
    if p >= 0.30:
        return "possible"
    if p >= 0.15:
        return "unlikely"
    return "very unlikely"
