# Context for writing the report: Betfair greyhound market maker (FNCE30010 backtesting assignment)

You are helping write the report for a university backtesting assignment (FNCE30010, authors Lawrence Hider and Warren Kusuma). This file summarises everything built and found so far. Treat the numbers here as the source of truth. If something is marked pending, don't invent a result for it.

**How we want the writing:** clear, concise, plain English that sounds human, not AI-generated. Explain each method intuitively before any formula. Australian/British spelling (favour, colour, behaviour). Don't refer to "the guide" or any step-by-step instructions document: present every method choice as our own decision with its reason. Avoid padding the report with extra checks that aren't needed to support a claim.

---

## 1. What the strategy is

A market maker on Betfair Australian greyhound **WIN** markets. For each runner priced between odds 1.2 and 6.0, it rests one back and one lay order from **350 s before the scheduled off** until **T = 30 s before it**, quoting **10 contracts** per side (a contract pays $1 if the runner wins, so stake = 10 / odds). Position cap: **50 contracts** either way per runner.

**Prices are modelled in log-odds.** For decimal odds O, a price's implied probability is 1/O and its log-odds is x = −ln(O − 1). Mid: S = ½(1/B + 1/L), with B the best back odds and L the best lay odds, and x₀ = ln(S/(1−S)). Log-odds removes the 0–1 bounds of probability, so moves are comparable for favourites and longer shots.

**Sides.** Our back is the bid (below the mid in probability, filled by layers). Our lay is the ask (above the mid, filled by backers).

**Avellaneda–Stoikov (A–S) quotes**, computed in probability units:
- reservation price r = S − qγσ²(T − t), which leans the quotes against our inventory q;
- half-spread δ = ½γσ²(T − t) + (1/γ)·ln(1 + γ/k), a risk term plus a fill term;
- σ² and k come from the tables below. They are converted from log-odds to probability with S(1−S) (σ²_p = σ²_x·[S(1−S)]², k_p = k_x / [S(1−S)]) and linearly interpolated between bin midpoints.

**Glosten–Milgrom adverse-selection premium.** Each side's A–S quote is widened by â, the markout a LightGBM model predicts for a fill at that quote. It is clipped to [0, 0.15] in log-odds and converted with S(1−S). The model describes the A–S quote *before* the premium is added (otherwise the premium would move the quote it is predicting from).

**Green-up.** At T the strategy cancels its quotes and crosses the spread (up to 3 ticks of slippage) to flatten every position of **at least 1 contract**. Smaller positions are left to settle on the race result.

**Diming (toggle, `dime=True`).** A quote that lands inside the spread may go at most one tick in front of the best price on its own side, so it beats the next price in the queue by exactly one tick. Default: off. Without diming, the anti-crossing rule puts a would-cross quote one tick short of crossing, which can be several ticks in front of everyone else when the spread is wide.

**Final settings:** γ = 0.5, greenup_min_units = 1, diming off by default. Code: `strategies/_final-strategy.py`, class `ASGMStrategy`.

## 2. Backtest engine and data

- Data: Betfair PRO stream for every 2025 Australian greyhound WIN market (**47,458 races**), with the best 3 prices and sizes on each side, and amounts traded since the previous update (halved, because Betfair counts each matched bet on both sides).
- Engine (`backtest/execution.py`):
  - **0.2 s latency** on every order and cancel.
  - A resting order fills when it is **crossed** (the other side moves onto its price), **traded through** (a trade at a worse price for the counterparty), or when **trades at its price use up the queue ahead of it**. The queue is the visible size at that price on arrival, shrinks when the visible size drops, and is unknown (treated as infinite) beyond the 3 visible levels.
  - Everything lapses when the market suspends.
- Commission: **4% of each market's net winnings**. Bets on scratched runners are voided, and prices are adjusted by Betfair's reduction factors.

**Data split:**

| Period | Used for |
|---|---|
| January 2025 | estimating σ², k and the markout horizon; training the adverse-selection model (train 1–25 Jan, validate 26–31 Jan, refit on all of January) |
| 1–14 February | tuning γ and the green-up threshold |
| 15 February – 31 December | out-of-sample evaluation only (nothing fitted or tuned here) |

Time bins (seconds before the off), used throughout: (30,45], (45,60], (60,90], (90,120], (120,180], (180,240], (240,300], (300,350]. Confidence intervals are bootstraps that resample **whole races** (1,000 draws), because runners in a race move together.

## 3. Volatility σ² (`research/volatility.ipynb`)

**Method.** Sample each runner's mid every 5 s from 350 s to 30 s before the off (the latest book update at or before each time). Take 5-second changes in log-odds, keeping a change only if the odds were in 1.2–6.0 at its start. Within each bin, σ² = variance of the changes ÷ 5.

| Bin | 30–45 | 45–60 | 60–90 | 90–120 | 120–180 | 180–240 | 240–300 | 300–350 |
|---|---|---|---|---|---|---|---|---|
| σ² (×10⁻⁴ per s) | 2.14 | 2.47 | 2.05 | 1.69 | 1.69 | 1.76 | 1.82 | 1.40 |

- Quietest early (1.40) and busiest 45–60 s before the off (2.47, about 1.8×). Over a minute, σ² = 2×10⁻⁴ means log-odds typically move about 0.11, roughly 2.8–3.2 for a dog at 3.0.
- Every 95% interval is within about 10% of its estimate. Day to day, a typical day is within 20–35% (daily CV 0.20–0.35), noisiest near the off.
- Requiring odds to finish in range as well would understate σ² by 2–8%. Dropping moves larger than 1 in log-odds (almost all from one illiquid race) changes it by under 5%.
- **Why 5-second steps.** Five seconds is the longest step that divides every bin edge. A volatility signature plot (variance per second against step length, 1–80 s) shows that:
  - 1-second steps overstate σ² by about 6% overall, and by about 30% (95% CI 21–40%) in the earliest bin: the mid flickers, moving and moving straight back;
  - anything from 4 to 16 s agrees with 5 s within 2%;
  - longer steps creep up 3–5%, a little drift over minutes, with wide intervals.
  It also matches the 5-second windows used for k.

## 4. Fill rate k (`research/fill_rate.ipynb`)

**Model:** λ(δ) = A·e^(−kδ), the fills per second for a quote at distance δ (log-odds) from the mid. Only k enters the A–S formulas.

**Method:**
1. Every 5 s from 350 s to 35 s, place pretend back and lay quotes at every ladder price out to 0.30 from the mid. Each arrives after the 0.2 s latency and lives 5 s.
2. Count a quote as filled **under the engine's own rules**: crossed, traded through, or queue used up, with the queue modelled. A further-out quote can only fill if the nearer ones do, so each window reduces to the distance of the furthest quote filled.
3. P(δ) = share of windows filled at δ (the first ladder price at least δ away), turned into a rate with λ = −ln(1 − P)/5 (Poisson arrivals).
4. Ordinary least squares of ln λ on δ, using distances with 2% < P < 98%. k = −slope.
5. A rule merges any bin with k < 2 or fewer than 3 usable distances into a neighbour. It wasn't needed for January.

| Bin | 30–45 | 45–60 | 60–90 | 90–120 | 120–180 | 180–240 | 240–300 | 300–350 |
|---|---|---|---|---|---|---|---|---|
| k, back | 47.4 | 38.1 | 38.7 | 36.6 | 32.9 | 27.2 | 20.0 | 14.4 |
| k, lay | 43.3 | 40.3 | 37.4 | 36.7 | 30.5 | 23.8 | 17.1 | 11.6 |

- k rises three- to fourfold towards the off. One tick (about 0.02 log-odds) further out keeps e^(−0.02k) of the fills: 76% at k = 14, but 39% at k = 47.
- Fit: R² ≥ 0.993 in every bin. Every 95% interval is within about 7% of its estimate. Daily CV 0.08–0.15 (0.20–0.24 in the earliest bin): steadier than σ².
- **Queue position.** An earlier version assumed we were always first in line. Modelling the queue lowered k near the off (lay 47.5 → 43.3, back 49.4 → 47.4) and raised it slightly early on (crossing fills that a trade-only rule missed). Remaining limitation: like the engine, we assume people ahead of us in the queue cancel first, which is slightly optimistic.
- **Back vs lay.** From 120 s out, the lay's k is 8–24% smaller than the back's (by 2.4–3.5), and a paired bootstrap puts every gap clear of zero. In the last two minutes there's no consistent direction. A lay 0.01 from the mid is filled 10–50% more often than a back. Interpretation: lays are filled by backers, who arrive more often and, early on, trade further from the mid (plausibly punters backing at whatever price is on offer). Effect: far from the off the strategy quotes its lay slightly wider, by about 0.005–0.017 in log-odds (up to about a tick).
- **Spread check.** The market's median half-spread is about 0.02 near the off and 0.12 at 300–350 s (13 ticks). 1/k is close to the half-spread in the last two minutes and well inside it earlier.

## 5. Adverse selection (`adverse-selection/`)

### 5a. Markout horizon (`choosing-markout-horizon.ipynb`)
- **Pretend quotes** at every ladder price within 0.20 of the mid, 10 contracts each:
  - posted at 350 s and at every change in the best back or lay price;
  - cancelled at the next change, with the 0.2 s cancel latency (so pick-offs count);
  - filled by the engine's rules, including partial fills. A check against the real engine matched 1,800 of 1,805 fills.
  - In total: 25.9 million pretend quotes, of which 929,881 were at least partly filled.
- **Markout** = the move in the mid against us, from the mid when the quote was posted (x₀) to h seconds after the fill, capped at T. Runner events (|move| > 1) are dropped.
- **Findings:**
  - A bounce at h = 0 (0.048, falling to 0.034 at 0.5 s): temporary pressure from the trade itself.
  - Then a fast rise that flattens: 0.047 at 10 s, 0.053 at 30 s, about 0.056 by 60 s.
  - The noise keeps growing: the standard deviation is 0.055 at 1 s, 0.102 at 30 s and 0.181 at 120 s.
  - The closer to the off, the sooner the curve flattens. Fills 300–350 s out are still drifting at 120 s, but they are under 7% of filled contracts.
- **Chosen h = 30 s**: the shortest horizon from which the markout stays at or above 90% of its 120 s level, at the bootstrap lower bound (25 s just misses at 89%).

### 5b. Features (`building-features.ipynb`)
Sixteen features, all from the order book **when the quote was posted** (never just before the fill, which would leak the move into the target). Directional features are signed so that positive means bad for our quote. All are rounded to 9 decimals so float noise can't flip a tree split.
- **Features:** side, quote_distance, ticks_from_best, seconds_to_off, mid_odds, favourite, spread_ticks, half_spread, imbalance_1, imbalance_3, move_5, move_30, vol_30 (std of 1-s mid changes over 30 s), traded_10, traded_30, flow_30.
- **Data:** 928,814 filled pretend quotes from January, weighted by contracts filled.
- **Patterns, one feature at a time:**
  - Markout grows with quote distance (about 0.02 for the closest tenth of quotes, 0.09 for the widest) and with time to the off (about 0.026 near it, 0.064 at 300 s or more).
  - Wide spreads, little trading and very low volatility mean bigger markouts: thin early markets.
  - Recent moves and volatility are U-shaped.
  - Order flow against the quote raises markouts.

### 5c. Model (`training-the-model.ipynb`)
- **Setup:** LightGBM regression (num_leaves 15, min_child_samples 200, learning rate 0.05, seed 42), weighted by contracts. Early stopping on 26–31 Jan picked **220 trees** (121 trees get within 0.1% of that error). Refit on all of January and saved as `adverse_model.txt`.
- **Accuracy on 26–31 January (unseen):**

  | Model | Share of variation explained |
  |---|---|
  | One average | 0% |
  | Lookup table (time bin × side) | 2.4% |
  | Straight line on 16 features | 6.4% |
  | **LightGBM** | **8.75%** |

- **Against the lookup table:** LightGBM's error is **6.5% lower (95% CI 5.3–7.7%)**, and lower on **all six** validation days separately (by 3.9–10.5%). Our rule was to use the model only if it beats the table; it passes.
- **Calibration** is good, slightly high at the top (predicts 0.099 against an actual 0.092, the safer direction for a premium).
- **It separates bad fills from good.** Within each distance band, net per contract (distance − markout) falls steadily from the safest predicted fifth to the most dangerous. For example, at 0.05–0.10 from the mid it goes from +0.033 to −0.017; at 0.025–0.05, from +0.007 to −0.034.
- **What it relies on** (SHAP): quote_distance and seconds_to_off most, then imbalance_3, ticks_from_best, flow_30 and move_5/move_30. favourite adds almost nothing.
- **Speed:** about 50 µs for a 10-quote prediction, far shorter than the time between book updates.

## 6. Tuning γ and the green-up threshold (`research/gamma_grid.ipynb`)

- **Grid:** γ ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3}, a roughly log-spaced grid shown on log axes. Threshold ∈ {0, 1, 2.5, 5, 10, 20, never}, in contracts. Run on 1–14 February.
- **One backtest per γ**, with the threshold at 0. Other thresholds were computed by dropping the green-up bets a higher threshold would skip, then resettling each market with commission. This reproduces each run exactly at threshold 0. Against two real check backtests it differs in 3–6 races, where skipping green-ups shifts later fills or a race is put back after T.
- **Profit by γ** (threshold 1, 1–14 Feb, after commission):

  | γ | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.75 | 1 | 1.5 | 2 | 3 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | Profit ($) | 69 | 87 | 88 | 103 | 105 | **110** | 101 | 83 | 75 | 62 | 53 |

  - **γ = 0.5** is the peak, with a 95% CI of $68–147. Values from 0.3 to 0.75 are statistically tied with it (paired intervals span zero).
  - Every γ ≥ 1 is significantly worse (paired intervals below zero): it quotes so wide it barely trades, about half as many races at γ = 3.
  - Risk per race falls steadily as γ rises (std of race profit 0.83 at γ = 0.05, 0.51 at 0.5, 0.21 at 3). Daily Sharpe is about 1.1–1.4 for every γ ≥ 0.1, flat within the noise of 15 days, so the profit peak decides.
- **Threshold:**
  - **1 contract** beats flattening everything for **all 11 γ** without adding risk: these are tiny positions where flattening crosses the spread to remove almost no risk.
  - **2.5** beats it for 9 of 11 γ.
  - **From 5 upwards** it's a lottery: 4–6 of 11 γ, the std of race profit is 6–11× larger, and the intervals run to hundreds of dollars either side of zero. The +$250–370 cells at thresholds 10–20 are race luck (holding positions is a bet on the result).
- **Real backtest with γ = 0.5 and threshold 1: +$106.87** after commission over 1–14 February (against $104.21 flattening everything), profitable on 13 of 15 racing days. The worst day was −$3.40, and the largest drawdown $8.53.

## 7. Diming

Implemented and tested:
- with it off, the results are identical to the previous version;
- with it on, none of 149k posted quotes was more than one tick in front, and none crossed;
- without it, 9.6% of quotes sat more than one tick in front, by up to 22 ticks.

Diming also matches the k estimate's first-in-line assumption, because a dimed quote inside the spread is always first in line. **The full-year comparison is pending.**

## 8. Evaluation (pending: full-year 2025 backtests are running)

- `evaluation/market-behaviour.ipynb`: random out-of-sample races (15 Feb – 31 Dec), plotted per runner. It shows the market bid–ask, the mid, our bid and ask, our fills, others' trades, and our position. What it shows:
  - our quotes start well outside the market's best prices, because the risk term and the premium are large early, and close in towards the off;
  - fills come when others' trades reach our price;
  - after a fill, both quotes shift to work inventory back to zero;
  - positions end flat at the off;
  - gaps in our quotes appear when a runner drifts beyond odds of 6.0 or the position cap binds.
- `evaluation/strategy-evaluation.ipynb` (results pending), out-of-sample only:
  - profit with bootstrap CI, per race, profitable days, daily and annualised Sharpe, largest drawdown and fills;
  - cumulative and monthly profit;
  - a paired race-by-race comparison of diming against no diming.
  - Planned: a P&L decomposition (spread captured, adverse move, green-up cost, settlement), profit by time to the off, odds and distance, and inventory behaviour.
- Results folders: `research/results/final_2025_no_dime/` and `research/results/final_2025_dime/`.

## 9. Limitations worth discussing in the report

- **Backtest realism:**
  - Our orders don't affect the historical book or other traders.
  - The queue is inferred from visible size, assuming cancellations come from ahead of us.
  - Beyond 3 price levels the book is unseen.
  - Minimum bet sizes aren't modelled.
- **Parameters fitted on one month (January)** and assumed stable for the year. There was day-to-day variation: CV 0.2–0.35 for σ², 0.08–0.15 for k.
- **Single k per bin averages across runners.** Far from the off, short-priced runners have a higher k than longer-priced ones.
- **The A–S model assumes a random walk and Poisson fills.** There's slight drift at horizons over a minute, and k depends on window length (longer windows give a lower k near the off), so short windows were used to match a strategy that re-centres its quotes at every price change.
- **The markout target is noisy.** The model explains about 9% of the variation: useful for ranking fills, not for precise prediction.
- **Tuning used only two weeks.** The γ peak is broad, and the threshold choice rests on consistency across γ rather than on its small dollar gain.

## 10. Where things are

| File | What it is |
|---|---|
| `strategies/_final-strategy.py` | final strategy (`ASGMStrategy`), defaults = final settings |
| `research/volatility.ipynb` | σ² estimation and the 5-second justification |
| `research/fill_rate.ipynb` | k estimation (queue-aware), back vs lay |
| `adverse-selection/choosing-markout-horizon.ipynb` | pretend quotes, markouts, h = 30 s |
| `adverse-selection/building-features.ipynb` | the 16 features |
| `adverse-selection/training-the-model.ipynb` | LightGBM vs benchmarks, calibration, SHAP |
| `research/gamma_grid.ipynb` | tuning γ and the green-up threshold |
| `evaluation/market-behaviour.ipynb` | per-race behaviour charts |
| `evaluation/strategy-evaluation.ipynb` | full-year evaluation (pending results) |
| `backtest/execution.py` | engine fill rules (latency, queue, crossing) |
| `scripts/run_backtest.py` | backtest runner |
