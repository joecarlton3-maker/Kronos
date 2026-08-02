# Adaptive Volatility SuperTrend + MACD

A trend-continuation system for crypto/24-7 markets. The **adaptive SuperTrend**
is used as a *regime filter and trailing exit* — not as the entry trigger — while
a **MACD cross** times the entry. The SuperTrend's ATR multiplier adapts to the
volatility regime and trend efficiency: wider bands in choppy / high-relative-vol
conditions (fewer whipsaws), tighter in clean trends (locks gains).

Two files, both PineScript **v6**:

| File | Type | Use |
|------|------|-----|
| `adaptive_volatility_supertrend.pine` | `indicator` | Chart visuals + alerts |
| `adaptive_volatility_supertrend_strategy.pine` | `strategy` | Backtesting in the Strategy Tester |

## Execution rules

The adaptive SuperTrend is a **regime filter and trailing exit, not the sole entry trigger.**

| Component | Long | Short |
|---|---|---|
| HTF permission | 1h adaptive SuperTrend bullish | 1h adaptive SuperTrend bearish |
| Entry | 15m MACD bullish cross while 15m SuperTrend is bullish | 15m MACD bearish cross while 15m SuperTrend is bearish |
| Initial stop | Below the adaptive SuperTrend **or** 1.2–1.5×ATR from entry, **whichever is farther** | Above the SuperTrend or 1.2–1.5×ATR from entry, whichever is farther |
| Trade management | Partial at 1–1.5R; trail the remainder with the adaptive SuperTrend | Same |
| No-trade | SuperTrend direction flips repeatedly within the last 6–10 bars | Same |

Key behaviors baked into the code:

- **Completed-candle signals.** Everything is gated on `barstate.isconfirmed`;
  the strategy runs `process_orders_on_close=true` + `calc_on_every_tick=false`.
  Intrabar MACD crosses that disappear before the bar closes are ignored — the
  right call for a crossover system.
- **Initial stop = the *farther* of ST-line vs ATR**, captured at entry. That is
  the stop that gives the trade the most room ("more protective of the trade
  structure"), not the tightest one.
- **Ratchet-only stop.** After entry the stop moves in the favorable direction
  only (up for longs, down for shorts), trailing the adaptive SuperTrend line.
- **Partial then trail.** At the first target it exits the configured partial
  (default 50%) and leaves the remainder to follow the SuperTrend.
- **Chop guard.** If the SuperTrend flips `chopMax` times within `chopLen` bars,
  new entries are suppressed.

## Recommended starting configuration

| Component | Starting setting |
|---|---|
| Chart timeframe | 15 minutes |
| Higher-timeframe filter | 1 hour |
| Adaptive ATR length | 10 |
| Base multiplier | 2.8 |
| Adaptive multiplier range | 1.8 to 4.2 |
| MACD | 12, 26, 9 |
| Initial ATR-stop distance | 1.5 ATR |
| First take-profit | 1.5R |
| Partial exit size | 50% |
| Commission | Replace `0.05%` with your actual all-in fee estimate |

The chart/HTF timeframes are set on the chart and the `HTF Timeframe` input; the
rest are strategy inputs and already default to the values above.

## How the adaptive multiplier works

1. **Volatility regime** — ATR is normalized by price (`atr/close`) and compared
   to its long baseline (`ta.ema` over `volLookback`). The ratio drives the
   multiplier via `baseMult · volRatio^gamma`, clamped to `[minMult, maxMult]`.
2. **Trend efficiency** — Kaufman's Efficiency Ratio scales it by
   `erBase − erSlope·ER`: high ER (clean trend) tightens, low ER (chop) widens.
   Set `erSlope` negative if you'd rather *widen* in trends instead.
3. **Smoothing** — the final multiplier is EMA-smoothed (`smoothLen`).

## Tuning discipline

**Do not optimize every input at once.** Test these one at a time, holding the
rest fixed, with the *same commission, slippage, and date range* across every
comparison:

1. Base multiplier
2. Volatility elasticity (`gamma`)
3. ATR-stop multiple (`stopAtr`)
4. First take-profit (`tp1R`)

The goal is **lower drawdown and stability across regimes — not the highest
historical net profit.** Prefer a *plateau* of good results over a lone peak, and
track trade count at every grid point (a great PF on 15 trades is noise). Aim for
≥100 trades before trusting any number, and add realistic costs before drawing
conclusions — TradingView's zero-cost default flatters everything.

## Synthetic validation with Kronos

Real history is a single timeline; walk-forward still only tests regimes that
actually happened. **Kronos** (this repo) generates statistically realistic
synthetic K-lines, letting you replay the rules across many plausible paths and
read the *distribution* of PF / expectancy / max-DD instead of one number.

1. Condition Kronos on a representative slice of the target symbol
   (`examples/prediction_example.py`).
2. Sample N synthetic OHLCV sequences (50–500).
3. Port these exact entry/exit rules to Python (`vectorbt` / `backtesting.py`) and
   replay each — `examples/run_backtest_kronos.py` is the existing scaffold.
4. Inspect the outcome distribution. "Mean PF 1.6 with 90% of runs above 1.2" is
   very different from "mean PF 1.6 with 30% underwater," though a single-history
   backtest can't tell them apart.

Caveats: generators don't invent unseen tail events, and synthetic validation is
*additive* — it de-risks, it doesn't prove. Forward/paper results are the final
check.
