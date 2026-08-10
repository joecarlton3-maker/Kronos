# 13/48/200 EMA Fan + Key Levels Bias System

A TradingView indicator (PineScript v6) that encodes an intraday momentum
playbook for index ETFs ($SPY, $QQQ, $IWM): trade **with** the EMA fan, at
**key levels**, with a **checklist-driven bias** — one or two confluence trades
a day, base hits.

File: [`ema-fan-levels-bias.pine`](ema-fan-levels-bias.pine)

## Setup

1. Open TradingView → Pine Editor → paste the contents of the `.pine` file →
   **Add to chart**.
2. Use a **2m–15m intraday chart** (2m is the reference timeframe).
3. Turn **extended hours ON** (chart settings → Session → Extended trading
   hours) — pre-market high/low tracking needs it. Without it the PMH/PML rows
   show `—` and everything else still works.

## What it draws

| Element | Meaning |
|---|---|
| Yellow / purple / red lines | 13 / 48 / 200 EMAs — the fan |
| Blue lines with shaded bands | Previous Day High / Low **zones** (width configurable) |
| Gray dotted lines | Pre-Market High / Low |
| Checklist table | Live bullish-bias checklist + verdict |

PDH/PDL are the previous **cash session** (regular trading hours) high/low,
tracked directly from RTH bars — extended-hours prices never contaminate them,
whether or not the chart displays extended hours.

## The bias framework

- Price above the **200 EMA** → overall lean bullish. Below → lean bearish.
- Price above the **200 + 48** → focus on calls only.
- Price above **all three** (13+48+200) → bulls have full control. Calls only,
  no exceptions. Mirror logic for puts below all three.
- **No shorts above the 13/48 EMAs. No longs below them.** The table footer
  keeps this rule in your face.

The checklist scores four conditions (above PDH, above PMH, bullish EMA fan,
above 200 EMA) plus a bull-flaggy price-action row, and prints a verdict from
`FULL BULL — CALLS ONLY` down to `FULL BEAR — PUTS ONLY`.

## The signals

**Level plays** (PDH/PDL treated as zones, with a signal cooldown):

1. **Break & retest PDH holds** → bullish (`PDH B&R`)
2. **Reject at the PDH** → bearish (`PDH REJ`)
3. **Bounce at the PDL** → bullish (`PDL BNC`)
4. **Break & retest PDL fails** → bearish (`PDL B&R`)
5. **Opened above the PDH** → PDH flips to the day's main support; a dip into
   the zone that holds fires `PDH SUP` (bullish). The checklist row renames
   itself to "Above PDH — main support" on those days. Mirrored: opened below
   the PDL → `PDL RES` (bearish).
6. **Pre-market level breaks** (one-shot per day, RTH only): first close
   through the PMH → `PMH BRK` (calls trigger); first close through the PML →
   `PML BRK` (puts trigger). Breaks require a **close** through the level —
   a wick through it is not a break.
7. **PMH–PML chop filter**: "avoid trading anything between those 2 levels."
   While price sits inside the pre-market range, EMA dip and flag entry
   signals are muted (toggleable), the range is shaded, and the table footer
   switches to "Chop: inside PMH–PML — be patient." Level plays, PMH/PML
   breaks, and EMA cross / reversal signals still fire — those are the exits
   from the chop.

**Trend continuation** (only while the fan and 200 EMA agree):

- `13 DIP` — first pullback into the 13 EMA that holds → entry zone
- `RE-ADD` — second 13 EMA dip → add spot if it holds
- `48 DIP` — deeper pullback into the 48 EMA later in the trend
- `13 POP` — mirror signal: rally into the 13 EMA in a downtrend
- `FLAG` — micro bull/bear flag: 2+ lower-high pullback bars holding the fan,
  then a close through the prior high (mirrored for bear flags)

**Reversal confirmation:**

- `13x48` — 13/48 EMA cross (short-term trend shift)
- `REV` — a PDH rejection followed within 20 bars by a bearish 13/48 cross
  (or PDL bounce + bullish cross). Two confirmations, not one.

## Alerts

Create **one** alert: chart → Alert → Condition → this indicator → **"Any
alert() function call"**. Every signal then arrives with a descriptive message
prefixed by the ticker. Hiding a signal group in the settings also mutes its
alerts, and the whole thing can be switched off with the "Enable alert()
events" input. Signals only fire on **confirmed bar closes** and daily levels
use the prior completed session — the indicator does not repaint.

## Discipline rules baked into the design

- Multiple confirmations to stay with the trend; zero confirmations to fade it.
  Don't try to time the top.
- Take 1–2 confluence trades a day. Small size. Base hits.
- Trim into new highs, hold runners toward the PDH/PDL target zones.

## Disclaimer

Educational tooling, not financial advice. Test on paper before risking money.
