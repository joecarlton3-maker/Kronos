"""Pretty-print PathStats as a terminal-friendly futures decision report."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from .stats import PathStats


def format_report(
    stats: PathStats,
    symbol: str,
    timeframe: str,
    lookback_bars: int,
    generated_at: Optional[datetime] = None,
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
        "EXCURSION ENVELOPE (across full horizon)",
        sub,
        f"  Median max high:           {stats.median_max_high:>10.2f}  ({stats.median_max_high - stats.last_price:+.2f})",
        f"  Median min low:            {stats.median_min_low:>10.2f}  ({stats.median_min_low - stats.last_price:+.2f})",
        f"  90% of paths stay below:   {stats.p90_max_high:>10.2f}",
        f"  90% of paths stay above:   {stats.p10_min_low:>10.2f}",
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
        for t in stats.level_touches:
            tag = _touch_label(t.prob)
            lines.append(
                f"  {t.level:>10.2f} ({t.direction}):  {t.prob:>5.1%}  -> {tag}"
            )
        lines.append("")

    lines += [
        "TRADE PLAN (suggestions only - not orders)",
        sub,
        "  LONG:",
        f"    Stop  (P15 path low):    {stats.suggested_stop_long:>10.2f}  (risk {stats.last_price - stats.suggested_stop_long:.2f})",
        f"    Target (P75 path high):  {stats.suggested_target_long:>10.2f}  (reward {stats.suggested_target_long - stats.last_price:.2f})",
        f"    Reward:Risk:             {_fnum(stats.rr_long)} : 1",
        "  SHORT:",
        f"    Stop  (P85 path high):   {stats.suggested_stop_short:>10.2f}  (risk {stats.suggested_stop_short - stats.last_price:.2f})",
        f"    Target (P25 path low):   {stats.suggested_target_short:>10.2f}  (reward {stats.last_price - stats.suggested_target_short:.2f})",
        f"    Reward:Risk:             {_fnum(stats.rr_short)} : 1",
        "",
        "CAVEATS",
        sub,
        "  - Pretrained weights, not fine-tuned for your contract.",
        "  - Tail events under-represented; do not treat 0% as impossible.",
        "  - Re-run on regime shifts and avoid scheduled news windows.",
        "  - Decision support only - you make the trade.",
        sep,
    ]
    return "\n".join(lines)


def _pct(v: float) -> str:
    return f"{v:+.2f}%"


def _fnum(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.2f}"


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
