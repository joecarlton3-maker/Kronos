"""PNG fan chart export — visual companion to the trajectory bands.

Plots the recent close history alongside the median forecast and the
P10-P90 / P25-P75 percentile envelopes, plus optional plan and key levels.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .plan import PlanEvaluation
from .stats import PathStats


def fan_chart(
    stats: PathStats,
    recent_df: pd.DataFrame,
    output_path: Union[str, Path],
    *,
    symbol: str = "",
    timeframe: str = "",
    plan: Optional[PlanEvaluation] = None,
    levels: Optional[Sequence[Tuple[str, float]]] = None,
    history_bars: int = 100,
    figsize: Tuple[float, float] = (10.0, 6.0),
    dpi: int = 110,
) -> Path:
    """Render and save a fan chart PNG. Returns the output path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = recent_df.iloc[-history_bars:].reset_index(drop=True)
    h_x = np.arange(-len(history) + 1, 1)  # 0 = "now"
    f_x = np.arange(1, stats.horizon_bars + 1)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(h_x, history["close"].values, color="#1f1f1f", lw=1.2, label="History")

    p10 = np.array(stats.trajectory_close.p10)
    p25 = np.array(stats.trajectory_close.p25)
    p50 = np.array(stats.trajectory_close.p50)
    p75 = np.array(stats.trajectory_close.p75)
    p90 = np.array(stats.trajectory_close.p90)

    ax.fill_between(f_x, p10, p90, color="#3b82f6", alpha=0.15, label="P10-P90")
    ax.fill_between(f_x, p25, p75, color="#3b82f6", alpha=0.30, label="P25-P75")
    ax.plot(f_x, p50, color="#1d4ed8", lw=1.5, label="Median forecast")

    ax.axvline(0, color="#888", lw=0.8, ls="--")
    ax.axhline(stats.last_price, color="#444", lw=0.8, ls=":")

    for label, price in levels or []:
        ax.axhline(price, color="#9333ea", lw=0.7, ls="-.", alpha=0.7)
        ax.text(
            f_x[-1], price, f" {label}" if label else "",
            fontsize=8, color="#6b21a8", va="center",
        )

    if plan is not None:
        for color, price, lbl in [
            ("#15803d", plan.target, f"Target {plan.target:.2f}"),
            ("#b91c1c", plan.stop, f"Stop {plan.stop:.2f}"),
            ("#1f2937", plan.entry, f"Entry {plan.entry:.2f}"),
        ]:
            ax.axhline(price, color=color, lw=1.0, alpha=0.7)
            ax.text(
                h_x[0], price, f" {lbl}", fontsize=8, color=color, va="bottom",
            )

    title_bits = ["Kronos forecast"]
    if symbol:
        title_bits.append(symbol)
    if timeframe:
        title_bits.append(timeframe)
    title_bits.append(f"horizon={stats.horizon_bars}")
    title_bits.append(f"samples={stats.sample_count}")
    title_bits.append(f"P(up)={stats.p_up_terminal:.0%}")
    ax.set_title(" | ".join(title_bits), fontsize=11)
    ax.set_xlabel("Bars from now")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
