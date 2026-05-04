"""Bar fetchers — pluggable adapters that return OHLCV DataFrames.

The watch loop and CLI talk to this interface, not directly to a data
source. The repo ships:

  - CSVBarFetcher          static file (testing, backtests)
  - SubprocessBarFetcher   shell out to any user-provided script
  - TradingViewMCPFetcher  documentation/stub: subprocess wrapper
                           designed for a TradingView MCP CLI

We intentionally do not bundle a TradingView client. There is no
official TradingView MCP server, and unofficial ones violate TV's ToS
in various ways. The user wires their preferred data source by
configuring `SubprocessBarFetcher` with a command that prints OHLCV
CSV to stdout.

A bar fetcher's `fetch_bars(symbol, timeframe, lookback)` must return a
DataFrame with columns: timestamps, open, high, low, close. volume and
amount are optional.
"""
from __future__ import annotations

import io
import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLS = ["timestamps", "open", "high", "low", "close"]


class BarFetcher(ABC):
    @abstractmethod
    def fetch_bars(
        self, symbol: str, timeframe: str, lookback: int
    ) -> pd.DataFrame: ...


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Fetched bars missing required columns: {missing}")
    df = df.copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    return df


class CSVBarFetcher(BarFetcher):
    """Reads bars from a static CSV. Useful for backtests, tests, and
    the offline path of the watch loop (rotating CSV updated by an
    external process)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch_bars(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        df = _validate(df)
        if lookback > 0:
            df = df.iloc[-lookback:].reset_index(drop=True)
        return df


class SubprocessBarFetcher(BarFetcher):
    """Runs a user-supplied command, parses its stdout as CSV.

    The command receives three substituted placeholders: {symbol},
    {timeframe}, and {lookback}. Example:

        SubprocessBarFetcher(
            "tv-mcp-cli get-bars --symbol {symbol} --tf {timeframe} --n {lookback}"
        )

    The command's stdout must be CSV with at least
    timestamps,open,high,low,close columns.
    """

    def __init__(self, command_template: str, timeout: float = 30.0):
        self.command_template = command_template
        self.timeout = timeout

    def fetch_bars(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        cmd = self.command_template.format(
            symbol=symbol, timeframe=timeframe, lookback=lookback
        )
        result = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Bar fetcher command failed (rc={result.returncode}): {result.stderr.strip()}"
            )
        df = pd.read_csv(io.StringIO(result.stdout))
        df = _validate(df)
        if lookback > 0 and len(df) > lookback:
            df = df.iloc[-lookback:].reset_index(drop=True)
        return df


class TradingViewMCPFetcher(SubprocessBarFetcher):
    """Convenience wrapper for users who have a TradingView MCP server
    exposed as a CLI command. Defaults assume the command takes
    --symbol/--timeframe/--lookback flags and returns CSV on stdout.

    If your TV-MCP integration looks different, use SubprocessBarFetcher
    directly with your own template string."""

    DEFAULT_TEMPLATE = (
        "tradingview-mcp get-bars --symbol {symbol} "
        "--timeframe {timeframe} --bars {lookback}"
    )

    def __init__(self, command_template: Optional[str] = None, timeout: float = 30.0):
        super().__init__(command_template or self.DEFAULT_TEMPLATE, timeout=timeout)


def make_fetcher(spec: str) -> BarFetcher:
    """Tiny factory.

    spec formats:
      csv:/path/to/file.csv
      subprocess:tv-mcp-cli get-bars --symbol {symbol} ...
      tradingview                          (uses default template)
      tradingview:<custom command template>
    """
    if spec.startswith("csv:"):
        return CSVBarFetcher(spec.split(":", 1)[1])
    if spec.startswith("subprocess:"):
        return SubprocessBarFetcher(spec.split(":", 1)[1])
    if spec == "tradingview":
        return TradingViewMCPFetcher()
    if spec.startswith("tradingview:"):
        return TradingViewMCPFetcher(spec.split(":", 1)[1])
    raise ValueError(
        f"Unknown fetcher spec {spec!r}. "
        "Use csv:PATH, subprocess:CMD, tradingview, or tradingview:CMD."
    )
