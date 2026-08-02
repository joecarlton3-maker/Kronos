# Adaptive Volatility SuperTrend

A SuperTrend whose ATR multiplier **adapts to the volatility regime and trend
efficiency** instead of using a single fixed multiplier. Wider bands when
relative volatility is high or the tape is choppy (fewer whipsaws); tighter
bands in clean, efficient trends (locks gains).

Two files:

| File | Type | Use |
|------|------|-----|
| `adaptive_volatility_supertrend.pine` | `indicator` | Visual + alerts on a chart |
| `adaptive_volatility_supertrend_strategy.pine` | `strategy` | Backtesting in the Strategy Tester |

Both are PineScript **v6**.

## How the adaptive multiplier works

1. **Volatility regime** — ATR is normalized by price (`atr/close`) and compared
   to its own long baseline (`ta.ema` over `volLookback`). The ratio drives the
   multiplier through `baseMult * volRatio^gamma`, clamped to `[minMult, maxMult]`.
2. **Trend efficiency** — Kaufman's Efficiency Ratio (net move ÷ path length)
   scales the multiplier by `erBase - erSlope*ER`. High ER (clean trend) → tighter
   band; low ER (chop) → wider band.
3. **Smoothing** — the final multiplier is EMA-smoothed (`smoothLen`) so the band
   doesn't jitter bar to bar.

The SuperTrend then uses this per-bar multiplier instead of a constant.

## What changed vs. the original core

The original was an **indicator** (no entries/exits, not backtestable) with
repainting signals and hardcoded constants. Key improvements:

- **Ported to v6** and split into a clean indicator + a real, backtestable strategy.
- **Non-repainting signals.** Flip events are gated on `barstate.isconfirmed`;
  the strategy runs `process_orders_on_close=true` with `calc_on_every_tick=false`,
  so backtest fills are reproducible and match confirmed bars.
- **Every magic number is an input** (`erBase`, `erSlope`, band basis, etc.) with
  tooltips, grouped into logical sections.
- **Flip-confirmation buffer** (`flipBufAtr`) — require close to exceed the band
  by N ATRs before flipping, to cut whipsaws in ranges.
- **Guarded math** — no division-by-zero, no `na` propagation on warm-up bars.
- **Strategy adds** an optional HTF trend filter (non-repainting `request.security`),
  a protective ATR hard stop, an R-multiple take-profit, session gating + EOD
  flatten, and risk-based position sizing — all optional and off-by-default where
  they'd add curve-fitting surface.
- **Costs on by default.** Commission and slippage are set in the declaration;
  the defaults are placeholders — change them to your instrument/broker.

## Before you trust a backtest

This strategy is intentionally lean to resist overfitting, but the discipline
still applies:

- **Sample size** — ≥100 trades before any number means much.
- **Parameter sensitivity** — sweep `atrLen`, `baseMult`, `gamma`, `erSlope` at
  ±10% / ±25%. Look for a plateau of good results, not a lone peak.
- **Realistic costs** — the TradingView default of zero commission/slippage
  flatters everything. The declaration ships with costs on; tune them.
- **Walk-forward** — hold out the most recent ~25–30% and confirm it doesn't fall
  apart out of sample.

## Synthetic validation with Kronos

Real history is a single timeline — walk-forward still only tests the regimes
that actually happened. **Kronos** (this repo) generates statistically realistic
synthetic K-lines, letting you replay the strategy across many plausible paths
and look at the *distribution* of PF / expectancy / max-DD instead of one number.

Sketch:

1. Condition Kronos on a representative slice of the target symbol
   (see `examples/prediction_example.py`).
2. Sample N synthetic OHLCV sequences (50–500).
3. Port these entry/exit rules to Python (`vectorbt` / `backtesting.py`) and
   replay each sequence — see `examples/run_backtest_kronos.py` for the existing
   backtest scaffold.
4. Inspect the outcome distribution. "Mean PF 1.6 with 90% of runs above 1.2" is
   a very different result from "mean PF 1.6 with 30% underwater," even though a
   single-history backtest can't tell them apart.

Caveats: generators don't invent unseen tail events (a model that never saw a
2020-style crash won't produce one), and synthetic validation is *additive* —
it de-risks, it doesn't prove. Forward/paper results remain the final check.
