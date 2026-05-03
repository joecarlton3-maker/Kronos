"""Decision-support statistics computed from probabilistic forecast paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .trajectory import (
    DrawdownStats,
    TimeToTouch,
    TrajectoryBand,
    drawdown_long,
    drawdown_short,
    per_bar_band,
    time_to_touch,
)


# A level can be a bare price or a (label, price) tuple
LevelInput = Union[float, Tuple[str, float]]


@dataclass
class LevelTouch:
    level: float
    direction: str
    prob: float
    label: str = ""


@dataclass
class PathStats:
    last_price: float
    horizon_bars: int
    sample_count: int

    p_up_terminal: float
    p_up_quartiles: List[float]
    median_terminal: float
    mean_terminal: float
    median_terminal_pct: float

    terminal_p10: float
    terminal_p25: float
    terminal_p50: float
    terminal_p75: float
    terminal_p90: float
    iqr: float

    median_max_high: float
    median_min_low: float
    p90_max_high: float
    p10_min_low: float

    terminal_std: float
    terminal_std_pct: float
    atr_recent: float
    dispersion_in_atr: float

    pct_trend_up: float
    pct_trend_down: float
    pct_mean_revert: float
    pct_chop: float

    level_touches: List[LevelTouch]

    suggested_stop_long: float
    suggested_target_long: float
    suggested_stop_short: float
    suggested_target_short: float
    rr_long: float
    rr_short: float

    # Trajectory / drawdown / time-to-touch additions
    trajectory_close: TrajectoryBand
    trajectory_high: TrajectoryBand
    trajectory_low: TrajectoryBand

    drawdown_long: DrawdownStats
    drawdown_short: DrawdownStats

    time_to_target_long: TimeToTouch
    time_to_stop_long: TimeToTouch
    time_to_target_short: TimeToTouch
    time_to_stop_short: TimeToTouch

    level_time_to_touch: List[TimeToTouch] = field(default_factory=list)


def _normalize_levels(
    levels: Optional[Sequence[LevelInput]],
) -> List[Tuple[str, float]]:
    if not levels:
        return []
    out: List[Tuple[str, float]] = []
    for lvl in levels:
        if isinstance(lvl, (tuple, list)) and len(lvl) == 2:
            out.append((str(lvl[0]), float(lvl[1])))
        else:
            out.append(("", float(lvl)))
    return out


def compute_stats(
    paths: np.ndarray,
    recent_df: pd.DataFrame,
    levels: Optional[Sequence[LevelInput]] = None,
) -> PathStats:
    """
    Args:
        paths: (sample_count, pred_len, 6) — denormalized OHLCVA per path.
        recent_df: input DataFrame used as context for ATR.
        levels: list of price levels — each a float or (label, price) tuple.
    """
    levels_norm = _normalize_levels(levels)
    if paths.ndim != 3 or paths.shape[2] < 4:
        raise ValueError("paths must have shape (n_samples, pred_len, >=4)")
    n_samples, pred_len, _ = paths.shape

    highs = paths[:, :, 1]
    lows = paths[:, :, 2]
    closes = paths[:, :, 3]

    last_price = float(recent_df["close"].iloc[-1])

    terminal = closes[:, -1]
    p_up_terminal = float((terminal > last_price).mean())

    quart_idx = [max(1, int(pred_len * q)) - 1 for q in (0.25, 0.5, 0.75, 1.0)]
    p_up_quartiles = [float((closes[:, i] > last_price).mean()) for i in quart_idx]

    median_terminal = float(np.median(terminal))
    mean_terminal = float(terminal.mean())
    median_terminal_pct = (median_terminal - last_price) / last_price * 100

    pcts = np.percentile(terminal, [10, 25, 50, 75, 90])

    max_high_per_path = highs.max(axis=1)
    min_low_per_path = lows.min(axis=1)
    median_max_high = float(np.median(max_high_per_path))
    median_min_low = float(np.median(min_low_per_path))
    p90_max_high = float(np.percentile(max_high_per_path, 90))
    p10_min_low = float(np.percentile(min_low_per_path, 10))

    terminal_std = float(terminal.std())
    terminal_std_pct = terminal_std / last_price * 100 if last_price else float("nan")

    atr_recent = _atr(recent_df, 14)
    dispersion_in_atr = (
        terminal_std / atr_recent if atr_recent > 0 else float("nan")
    )

    shapes = [
        _classify_shape(closes[i], highs[i], lows[i]) for i in range(n_samples)
    ]
    counts = {
        k: shapes.count(k) / n_samples
        for k in ("trend_up", "trend_down", "mean_revert", "chop")
    }

    level_touches: List[LevelTouch] = []
    level_ttts: List[TimeToTouch] = []
    for label, lvl in levels_norm:
        if lvl > last_price:
            prob = float((max_high_per_path >= lvl).mean())
            direction = "above"
        else:
            prob = float((min_low_per_path <= lvl).mean())
            direction = "below"
        level_touches.append(
            LevelTouch(level=float(lvl), direction=direction, prob=prob, label=label)
        )
        level_ttts.append(time_to_touch(paths, lvl, last_price, label=label))

    stop_long = float(np.percentile(min_low_per_path, 15))
    target_long = float(np.percentile(max_high_per_path, 75))
    stop_short = float(np.percentile(max_high_per_path, 85))
    target_short = float(np.percentile(min_low_per_path, 25))

    risk_long = max(last_price - stop_long, 0.0)
    reward_long = max(target_long - last_price, 0.0)
    risk_short = max(stop_short - last_price, 0.0)
    reward_short = max(last_price - target_short, 0.0)
    rr_long = reward_long / risk_long if risk_long > 0 else float("nan")
    rr_short = reward_short / risk_short if risk_short > 0 else float("nan")

    trajectory_close = per_bar_band(closes)
    trajectory_high = per_bar_band(highs)
    trajectory_low = per_bar_band(lows)

    dd_long = drawdown_long(paths, last_price)
    dd_short = drawdown_short(paths, last_price)

    ttt_target_long = time_to_touch(paths, target_long, last_price, "Suggested target (long)")
    ttt_stop_long = time_to_touch(paths, stop_long, last_price, "Suggested stop (long)")
    ttt_target_short = time_to_touch(paths, target_short, last_price, "Suggested target (short)")
    ttt_stop_short = time_to_touch(paths, stop_short, last_price, "Suggested stop (short)")

    return PathStats(
        last_price=last_price,
        horizon_bars=pred_len,
        sample_count=n_samples,
        p_up_terminal=p_up_terminal,
        p_up_quartiles=p_up_quartiles,
        median_terminal=median_terminal,
        mean_terminal=mean_terminal,
        median_terminal_pct=median_terminal_pct,
        terminal_p10=float(pcts[0]),
        terminal_p25=float(pcts[1]),
        terminal_p50=float(pcts[2]),
        terminal_p75=float(pcts[3]),
        terminal_p90=float(pcts[4]),
        iqr=float(pcts[3] - pcts[1]),
        median_max_high=median_max_high,
        median_min_low=median_min_low,
        p90_max_high=p90_max_high,
        p10_min_low=p10_min_low,
        terminal_std=terminal_std,
        terminal_std_pct=terminal_std_pct,
        atr_recent=atr_recent,
        dispersion_in_atr=dispersion_in_atr,
        pct_trend_up=counts["trend_up"],
        pct_trend_down=counts["trend_down"],
        pct_mean_revert=counts["mean_revert"],
        pct_chop=counts["chop"],
        level_touches=level_touches,
        suggested_stop_long=stop_long,
        suggested_target_long=target_long,
        suggested_stop_short=stop_short,
        suggested_target_short=target_short,
        rr_long=rr_long,
        rr_short=rr_short,
        trajectory_close=trajectory_close,
        trajectory_high=trajectory_high,
        trajectory_low=trajectory_low,
        drawdown_long=dd_long,
        drawdown_short=dd_short,
        time_to_target_long=ttt_target_long,
        time_to_stop_long=ttt_stop_long,
        time_to_target_short=ttt_target_short,
        time_to_stop_short=ttt_stop_short,
        level_time_to_touch=level_ttts,
    )


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < 2:
        return float(df["high"].sub(df["low"]).mean()) if len(df) else 0.0
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    window = min(period, len(tr))
    return float(tr[-window:].mean())


def _classify_shape(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> str:
    net = close[-1] - close[0]
    rng = high.max() - low.min()
    if rng <= 0:
        return "chop"
    diffs = np.diff(close)
    total_path = np.abs(diffs).sum()
    efficiency = abs(net) / total_path if total_path > 0 else 0.0

    if efficiency > 0.5 and net > 0:
        return "trend_up"
    if efficiency > 0.5 and net < 0:
        return "trend_down"
    if abs(net) < 0.3 * rng:
        return "mean_revert"
    return "chop"
