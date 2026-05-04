"""Multi-timeframe confluence vote.

Given a list of (timeframe, PathStats) pairs, computes whether the
timeframes agree on direction, the conflict severity, and a single
aggregated bias. Useful for filtering trades to only those where the
1m / 5m / 15m views align.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .stats import PathStats


@dataclass
class TimeframeView:
    timeframe: str
    p_up: float
    bias: str               # 'long' / 'short' / 'neutral'
    horizon_bars: int


@dataclass
class ConfluenceResult:
    views: List[TimeframeView]
    aggregated_p_up: float
    aggregated_bias: str
    agreement_score: float       # 1.0 = all agree, 0.0 = perfect split
    conflict_flag: bool
    notes: List[str] = field(default_factory=list)


def _bias_from_p(p: float) -> str:
    if p >= 0.55:
        return "long"
    if p <= 0.45:
        return "short"
    return "neutral"


def multi_timeframe_vote(
    timeframe_stats: Sequence[Tuple[str, PathStats]],
) -> ConfluenceResult:
    if not timeframe_stats:
        raise ValueError("Need at least one (timeframe, stats) pair")

    views: List[TimeframeView] = []
    for tf, s in timeframe_stats:
        views.append(
            TimeframeView(
                timeframe=tf,
                p_up=float(s.p_up_terminal),
                bias=_bias_from_p(s.p_up_terminal),
                horizon_bars=int(s.horizon_bars),
            )
        )

    biases = [v.bias for v in views]
    longs = biases.count("long")
    shorts = biases.count("short")
    neutrals = biases.count("neutral")
    n = len(views)

    # Equal-weighted average P(up)
    agg_p_up = sum(v.p_up for v in views) / n
    agg_bias = _bias_from_p(agg_p_up)

    # Agreement score: fraction sharing the dominant non-neutral bias.
    # If all neutral, score = 1.0 (they agree, just no signal).
    if longs == 0 and shorts == 0:
        agreement = 1.0
    else:
        dominant = max(longs, shorts)
        agreement = dominant / n

    conflict = longs > 0 and shorts > 0

    notes: List[str] = []
    if conflict:
        notes.append("CONFLICT: timeframes disagree on direction. Stand aside.")
    elif neutrals == n:
        notes.append("All timeframes neutral. No signal.")
    elif longs == n:
        notes.append("All timeframes long. Strongest setup.")
    elif shorts == n:
        notes.append("All timeframes short. Strongest setup.")
    elif neutrals > 0 and not conflict:
        notes.append(
            f"{longs + shorts}/{n} timeframes directional, {neutrals} neutral. "
            "Acceptable but not unanimous."
        )

    return ConfluenceResult(
        views=views,
        aggregated_p_up=agg_p_up,
        aggregated_bias=agg_bias,
        agreement_score=agreement,
        conflict_flag=conflict,
        notes=notes,
    )


def format_confluence_report(res: ConfluenceResult) -> str:
    sep = "=" * 64
    sub = "-" * 64
    lines = [
        sep,
        "  KRONOS MULTI-TIMEFRAME CONFLUENCE",
        sep,
    ]
    lines.append(f"{'Timeframe':<12} {'Horizon':>10} {'P(up)':>10} {'Bias':>10}")
    lines.append(sub)
    for v in res.views:
        lines.append(
            f"{v.timeframe:<12} {v.horizon_bars:>10} {v.p_up:>10.1%} {v.bias.upper():>10}"
        )
    lines += [
        sub,
        f"  Aggregated P(up):     {res.aggregated_p_up:.1%}",
        f"  Aggregated bias:      {res.aggregated_bias.upper()}",
        f"  Agreement score:      {res.agreement_score:.0%}",
        f"  Conflict:             {'YES' if res.conflict_flag else 'no'}",
        "",
    ]
    for note in res.notes:
        lines.append(f"  - {note}")
    lines.append(sep)
    return "\n".join(lines)
