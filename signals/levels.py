"""Auto-derived key price levels from a recent OHLCV DataFrame.

Returns labeled levels traders actually care about (session VWAP, prior-day
H/L, prior close, current-session H/L) so the user doesn't have to type them.
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def auto_key_levels(df: pd.DataFrame) -> List[Tuple[str, float]]:
    """Compute key levels from `df`, which must contain 'timestamps',
    'high', 'low', 'close' (and ideally 'volume' for VWAP).

    Returns a list of (label, price) tuples. If a level cannot be computed
    (e.g. no prior session in the data), it is skipped.
    """
    if "timestamps" not in df.columns:
        raise ValueError("df must include a 'timestamps' column")
    df = df.copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    df["date"] = df["timestamps"].dt.date

    levels: List[Tuple[str, float]] = []
    today = df["date"].iloc[-1]
    today_bars = df[df["date"] == today]
    prior_bars = df[df["date"] < today]

    if len(today_bars) > 0:
        if "volume" in today_bars.columns and today_bars["volume"].sum() > 0:
            tp = (today_bars["high"] + today_bars["low"] + today_bars["close"]) / 3.0
            vwap = (tp * today_bars["volume"]).sum() / today_bars["volume"].sum()
            levels.append(("Session VWAP", float(vwap)))
        levels.append(("Today High", float(today_bars["high"].max())))
        levels.append(("Today Low", float(today_bars["low"].min())))

    if len(prior_bars) > 0:
        prior_day = prior_bars["date"].max()
        prior_day_bars = prior_bars[prior_bars["date"] == prior_day]
        levels.append(("Prior Day High", float(prior_day_bars["high"].max())))
        levels.append(("Prior Day Low", float(prior_day_bars["low"].min())))
        levels.append(("Prior Close", float(prior_day_bars["close"].iloc[-1])))

    return levels
