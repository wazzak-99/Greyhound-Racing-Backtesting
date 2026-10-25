"""
strategies/_final-strategy.py
-----------------------------
The final strategy: an Avellaneda-Stoikov market maker whose quotes are widened by a Glosten-Milgrom
adverse-selection charge, predicted for each quote by the LightGBM model in adverse-selection/.

`dime=True` stops a quote that lands inside the spread from going more than one tick in front of the
best price on its own side, so it only ever beats the next price in the queue by one tick.

Run from the project root (the file name has a hyphen, so pass the path, not a module name):

    python scripts/run_backtest.py --data Feb --strategy strategies/_final-strategy.py --class ASGMStrategy
    python scripts/run_backtest.py --data Feb --strategy strategies/_final-strategy.py --class ASGMStrategy \
        --params dime=True --out research/results/final-dime-feb
"""
from __future__ import annotations

import bisect
import math
from collections import deque
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from backtest.context import Cancel, MarketContext, MarketStatus, Order, RaceResult, RunnerState, Side
from backtest.execution import _TICK_BANDS, Fill, Unmatched
from backtest.strategy import Strategy


_EPS = 1e-9
INF = float("inf")

# ---------------------------------------------------------------------------
# Betfair price ladder
# ---------------------------------------------------------------------------

def _build_ladder() -> list[float]:
    """Builds the Betfair price ladder as a list of floats."""
    prices = []
    for low, high, step in _TICK_BANDS:
        for i in range(round((high - low) / step)):
            prices.append(round(low + i * step, 2))
    prices.append(1000.0)
    return prices

# The Betfair price ladder is a list of floats representing valid odds
# For example, you can't place an order at $2.35; the closest valid odds are $2.34 or $2.36
LADDER = _build_ladder()

def odds_round_up(odds: float) -> Optional[float]:
    """Smallest ladder price >= odds (None if above 1000)."""
    i = bisect.bisect_left(LADDER, odds - 1e-9)
    return LADDER[i] if i < len(LADDER) else None


def odds_round_down(odds: float) -> Optional[float]:
    """Largest ladder price <= odds (None if below 1.01)."""
    i = bisect.bisect_right(LADDER, odds + 1e-9) - 1
    return LADDER[i] if i >= 0 else None

def odds_tick(odds: float, n: int) -> Optional[float]:
    """Move `n` ticks along the ladder from a ladder price (`n` > 0 = longer odds, 
    `n` < 0 = shorter odds, otherwise None if invalid)."""
    i = bisect.bisect_left(LADDER, odds - 1e-9) + n
    return LADDER[i] if 0 <= i < len(LADDER) else None

def _ticks_apart(a: float, b: float) -> int:
    """Number of ladder ticks between two ladder prices."""
    return abs(bisect.bisect_left(LADDER, a - 1e-9) - bisect.bisect_left(LADDER, b - 1e-9))

def _log_odds(p: float) -> float:
    """Converts price (from 0 to 1) to log-odds (from -INF to INF)."""
    return math.log(p / (1 - p))

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------


K_BACK_X = [(45, 47.4), (60, 38.1), (90, 38.7), (120, 36.6), (180, 32.9), (240, 27.2), (300, 20.0), (INF, 14.4)]
K_LAY_X  = [(45, 43.3), (60, 40.3), (90, 37.4), (120, 36.7), (180, 30.5), (240, 23.8), (300, 17.1), (INF, 11.6)]
SIGMA2_X = [(45, 2.14e-04), (60, 2.47e-04), (90, 2.05e-04), (120, 1.69e-04), (180, 1.69e-04), (240, 1.76e-04), (300, 1.82e-04), (INF, 1.40e-04)]

# Type alias for A-S model parameters
# These can be given as a single value or a sequence of (time, value) tuples
Param = Union[float, Sequence[tuple[float, float]]]

def _lookup(param: Param, seconds_to_off: float) -> float:
    """Value of a constant or time-to-jump binned parameter."""
    if isinstance(param, (int, float)):
        return float(param)
    for max_seconds, value in param:
        if seconds_to_off <= max_seconds:
            return float(value)
    return float(param[-1][1])

def _interp(param: Param, seconds_to_off: float, lower_edge: float = 30.0) -> float:
    """Linear interpolation between bin midpoints; flat beyond the first and last midpoints."""
    if isinstance(param, (int, float)):
        return float(param)
    edges = [lower_edge] + [e for e, _ in param]
    edges[-1] = 350.0 if edges[-1] == INF else edges[-1]          # quoting window ends at 350 s
    mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    vals = [float(v) for _, v in param]
    if seconds_to_off <= mids[0]:
        return vals[0]
    if seconds_to_off >= mids[-1]:
        return vals[-1]
    i = bisect.bisect_right(mids, seconds_to_off) - 1
    w = (seconds_to_off - mids[i]) / (mids[i + 1] - mids[i])
    return vals[i] + w * (vals[i + 1] - vals[i])

# Parameters for the adverse selection (G-M) model
ADVERSE_X = [
    (45, [(INF, 0.0343)]),
    (60, [(INF, 0.0311)]),
    (90, [(INF, 0.0341)]),
    (120, [(INF, 0.0385)]),
    (180, [(INF, 0.0413)]),
    (240, [(INF, 0.0509)]),
    (300, [(INF, 0.0571)]),
    (INF, [(INF, 0.0674)]),
]

def _lookup_adverse(table, seconds_to_off: float, odds: float):
    """Value from a constant, a time-binned list, or a time x odds binned table."""
    if isinstance(table, (int, float)):
        return float(table)
    inner = table[-1][1]
    for max_seconds, value in table:
        if seconds_to_off <= max_seconds:
            inner = value
            break
    if isinstance(inner, (int, float)):
        return float(inner)
    for max_odds, value in inner:
        if odds <= max_odds:
            return value
    return inner[-1][1]

# ---------------------------------------------------------------------------
# Adverse-selection model (LightGBM)
# ---------------------------------------------------------------------------
# Trained in adverse-selection/training-the-model.ipynb on the 16 features from
# adverse-selection/building-features.ipynb. The features below must be computed exactly as there.

DEFAULT_ADVERSE_MODEL = Path(__file__).resolve().parents[1] / "adverse-selection" / "adverse_model.txt"
ADVERSE_FEATURES = ["side", "quote_distance", "ticks_from_best", "seconds_to_off", "mid_odds", "favourite",
                    "spread_ticks", "half_spread", "imbalance_1", "imbalance_3", "move_5", "move_30", "vol_30",
                    "traded_10", "traded_30", "flow_30"]
LOOKBACK_SECONDS = 30   # the features look at the last 30 s
FEATURE_DECIMALS = 9    # features are rounded as in training, so float noise (~1e-15) can't flip a tree's split

_MODELS: dict[str, object] = {}   # loaded once per process (the backtester makes a new strategy for every race)


def _load_adverse_model(path: str):
    if path not in _MODELS:
        import lightgbm
        # Read as text, which turns Windows line endings back into "\n": the file stores each tree's size
        # in bytes, so a checkout with CRLF endings can't be loaded with model_file=
        model = lightgbm.Booster(model_str=Path(path).read_text())
        if model.feature_name() != ADVERSE_FEATURES:
            raise ValueError(f"{path} expects features {model.feature_name()}, not {ADVERSE_FEATURES}")
        _MODELS[path] = model
    return _MODELS[path]


def _ladder_pos(price: float) -> int:
    """Position of a price on the ladder (1.01 is 0, 1.02 is 1, ...), so a difference counts ticks."""
    return bisect.bisect_left(LADDER, price - 1e-9)


def _mid_log_odds(r: RunnerState) -> float:
    """The runner's mid in log-odds (nan without both a back and a lay price)."""
    if not r.atb or not r.atl:
        return math.nan
    S = 0.5 * (1 / r.atb[0][0] + 1 / r.atl[0][0])
    return math.log(S / (1 - S))


class AdverseFeatures:
    """
    Keeps the last 30 s of each runner's mid and trading, and turns them into the model's 16 features,
    using only what is known at the current update.

    `update(ctx)` must be called on every snapshot, including those before the strategy starts quoting.
    """

    def __init__(self):
        # selection_id -> (time in ms, mid x, $ matched, $ matched above the mid, $ matched below), one per update
        self._history: dict[int, deque] = {}
        self._last_x: dict[int, float] = {}    # each runner's mid at its previous update

    def update(self, ctx: MarketContext) -> None:
        now = ctx.publish_time.value // 1_000_000
        for r in ctx.runners:
            sel = r.selection_id
            x = _mid_log_odds(r)
            before = self._last_x.get(sel, math.nan)   # trades are compared with the mid just before them
            history = self._history.setdefault(sel, deque())
            matched, above, below = history[-1][2:] if history else (0.0, 0.0, 0.0)
            for price, size in r.traded:
                amount = size / 2                      # Betfair counts each matched bet twice
                if not (price > 1.0) or not size or size <= 0:   # also catches NaN prices
                    continue
                trade_x = -math.log(price - 1)
                matched += amount
                if trade_x > before:                   # backing into layers' money: pushes the odds in
                    above += amount
                elif trade_x < before:                 # laying into backers' money: pushes the odds out
                    below += amount
            history.append((now, x, matched, above, below))
            while len(history) >= 2 and history[1][0] <= now - 1000 * LOOKBACK_SECONDS:
                history.popleft()                      # keep just one update from before the last 30 s
            self._last_x[sel] = x

    def runner_features(self, r: RunnerState, ttj: float, S: float, favourite: bool) -> dict:
        """The features that don't depend on our side or price. Directional ones are measured in the
        direction of the odds shortening; `quote_row` turns them into "against our quote"."""
        history = self._history[r.selection_id]
        now = history[-1][0]
        past = [None] * (LOOKBACK_SECONDS + 1)   # past[k] = the latest update at or before now - k seconds
        k = 0
        for entry in reversed(history):
            while k <= LOOKBACK_SECONDS and entry[0] <= now - 1000 * k:
                past[k] = entry
                k += 1
        x = [e[1] if e else math.nan for e in past]

        def added(column: int, seconds: int) -> float:
            # How much a running total grew over the last `seconds`
            return past[0][column] - past[seconds][column] if past[seconds] else math.nan

        changes = [a - b for a, b in zip(x[:-1], x[1:]) if not (math.isnan(a) or math.isnan(b))]
        if changes:
            mean = sum(changes) / len(changes)
            vol = math.sqrt(sum((c - mean) ** 2 for c in changes) / len(changes))
        else:
            vol = math.nan
        above, below = added(3, 30), added(4, 30)
        pushed = above + below
        flow = (above - below) / pushed if pushed > 0 else (0.0 if pushed == 0 else math.nan)

        B, L = r.atb[0][0], r.atl[0][0]                       # best back odds, best lay odds
        backers = [size for _, size in r.atl[:3]]            # money waiting to back (available to lay)
        layers = [size for _, size in r.atb[:3]]             # money waiting to lay (available to back)
        return {
            "x0": math.log(S / (1 - S)), "best_back": B, "best_lay": L, "seconds_to_off": ttj,
            "mid_odds": 1 / S, "favourite": 1.0 if favourite else 0.0,
            "spread_ticks": _ladder_pos(L) - _ladder_pos(B),
            "half_spread": 0.5 * (math.log(L - 1) - math.log(B - 1)),
            "imbalance_1": _share(backers[0], layers[0]), "imbalance_3": _share(sum(backers), sum(layers)),
            "move_5": x[0] - x[5], "move_30": x[0] - x[30], "vol_30": vol,
            "traded_10": added(2, 10), "traded_30": added(2, 30), "flow_30": flow,
        }

    @staticmethod
    def quote_row(f: dict, side: Side, odds: float) -> list[float]:
        """The model's 16 inputs for one quote, in ADVERSE_FEATURES order."""
        back = side is Side.BACK
        against = -1.0 if back else 1.0   # "odds shortening" -> "bad for our quote"
        ticks = (_ladder_pos(odds) - _ladder_pos(f["best_lay"]) if back
                 else _ladder_pos(f["best_back"]) - _ladder_pos(odds))
        return [
            1.0 if back else 0.0, abs(f["x0"] + math.log(odds - 1)), ticks, f["seconds_to_off"],
            f["mid_odds"], f["favourite"], f["spread_ticks"], f["half_spread"],
            against * f["imbalance_1"], against * f["imbalance_3"], against * f["move_5"], against * f["move_30"],
            f["vol_30"], f["traded_10"], f["traded_30"], against * f["flow_30"],
        ]

    @staticmethod
    def model_inputs(rows: list[list[float]]) -> np.ndarray:
        """Quote rows -> the array the model takes, rounded exactly as in building-features.ipynb."""
        return np.round(np.array(rows, dtype=float), FEATURE_DECIMALS)


def _share(up: float, down: float) -> float:
    """(up - down) / (up + down), or nan if both are zero."""
    total = up + down
    return (up - down) / total if total > 0 else math.nan

# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class ASGMStrategy(Strategy):
    def __init__(
        self,
        gamma: float = 0.5,                 # risk aversion, per $ of payout (tuned on 1-14 Feb, gamma_grid.ipynb)
        k_back_x: Param = K_BACK_X,         # fill decay of our resting BACKs (bid side), log-odds units
        k_lay_x: Param = K_LAY_X,           # fill decay of our resting LAYs (ask side), log-odds units
        sigma2_x: Param = SIGMA2_X,         # variance of log-odds per second
        quote_units: float = 10.0,           # payout units per quote ($ won if the dog wins)
        q_max: float = 50.0,                 # inventory cap, payout units
        start_seconds: float = 350.0,       # start quoting this many seconds before the off
        adverse_model: Optional[str] = str(DEFAULT_ADVERSE_MODEL),   # LightGBM model of the adverse move after
                                            # a fill (log-odds); None = use the ADVERSE_X table instead
        adverse_cap: float = 0.15,          # clip the model's prediction to [0, adverse_cap] (log-odds)
        adverse_x=ADVERSE_X,                # Glosten-Milgrom charge by time bin, used when adverse_model is None
        adverse_scale: float = 1.0,         # multiply the charge (1 = charge the full predicted move)
        adverse_keep_reducing: bool = False, # don't charge the side that reduces |q|
        greenup_seconds: float = 30.0,      # T_e: stop quoting and green up this many seconds before the off
        greenup_slippage_ticks: int = 3,    # how many ticks through the best price a green-up may take
        greenup_min_units: float = 1.0,     # leave |q| below this to settle (tuned on 1-14 Feb, gamma_grid.ipynb)
        min_odds: float = 1.2,              # only quote runners whose fair odds are in this range
        max_odds: float = 6.0,
        dime: bool = False,                 # inside the spread, go at most one tick in front of the best
                                            # price on our side (False = wherever A-S puts the quote)
    ):
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        self.gamma = gamma
        self.k_back_x = k_back_x
        self.k_lay_x = k_lay_x
        self.sigma2_x = sigma2_x
        self.quote_units = quote_units
        self.q_max = q_max
        self.start_seconds = start_seconds
        self.adverse_model = adverse_model
        self.adverse_cap = adverse_cap
        self.adverse_x = adverse_x
        self.adverse_scale = adverse_scale
        self.adverse_keep_reducing = adverse_keep_reducing
        self.greenup_seconds = greenup_seconds
        self.greenup_slippage_ticks = greenup_slippage_ticks
        self.greenup_min_units = greenup_min_units
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.dime = dime

        self.fill_log: list[dict] = []      # every fill across the run, for analysis afterwards
        self.market_log: list[dict] = []    # one row per runner per market

    # ---------------------------------------------------------------------------
    # Engine callbacks
    # ---------------------------------------------------------------------------

    def on_market_open(self, ctx: MarketContext) -> None:
        self.q: dict[int, float] = {}                         # selection_id -> inventory
        self.n_fills: dict[int, int] = {}
        self._quotes: dict[tuple[int, Side], Order] = {}      # our current resting quote per slot
        self._open: dict[int, list] = {}                      # id(order) -> [order, remaining stake]
        self._model = _load_adverse_model(self.adverse_model) if self.adverse_model else None
        self._features = AdverseFeatures() if self._model is not None else None

    def on_snapshot(self, ctx: MarketContext) -> list[Order | Cancel]:
        if self._features is not None:
            self._features.update(ctx)   # every snapshot, so the features always have the last 30 s
        if ctx.status is not MarketStatus.OPEN:
            return []
        ttj = ctx.seconds_to_off
        if ttj > self.start_seconds:
            return []
        if ttj <= self.greenup_seconds:
            return self._cancel_all() + self._green_up(ctx)

        fair = self._fair_values(ctx)
        actions: list[Order | Cancel] = []
        quotable: list[tuple[RunnerState, float]] = []
        for r in ctx.runners:
            S = fair.get(r.selection_id)
            if S is None or not (1 / self.max_odds <= S <= 1 / self.min_odds):
                actions += self._cancel_runner(r.selection_id)
                continue
            quotable.append((r, S))

        plain = {r.selection_id: self._as_prices(r, S, ttj) for r, S in quotable}   # A-S bid/ask, before any charge
        charges = self._adverse_charges(quotable, plain, fair, ttj)   # one model call for every runner and side
        for r, S in quotable:
            bid_odds, ask_odds = self._quote_odds(r, S, *plain[r.selection_id], *charges[r.selection_id])
            q = self.q.get(r.selection_id, 0.0)

            # Inventory cap: never post a quote whose full fill would take |q| past q_max
            back_units = min(self.quote_units, self.q_max - q)
            lay_units = min(self.quote_units, self.q_max + q)

            actions += self._requote(ctx, r.selection_id, Side.BACK, bid_odds, back_units)
            actions += self._requote(ctx, r.selection_id, Side.LAY, ask_odds, lay_units)
        return actions

    def on_fill(self, fill: Fill) -> None:
        order = fill.order
        sign = 1.0 if order.side is Side.BACK else -1.0
        sel = order.selection_id
        self.q[sel] = self.q.get(sel, 0.0) + sign * fill.stake * fill.price
        self.n_fills[sel] = self.n_fills.get(sel, 0) + 1

        entry = self._open.get(id(order))
        if entry is not None:
            entry[1] -= fill.stake
            if entry[1] <= _EPS:
                self._forget(order)

        self.fill_log.append({
            "market_id": order.market_id,
            "selection_id": sel,
            "publish_time": fill.publish_time,
            "side": order.side.value,
            "tag": order.tag,
            "price": fill.price,
            "stake": fill.stake,
            "q_after": self.q[sel],
        })

    def on_unmatched(self, unmatched: Unmatched) -> None:
        self._forget(unmatched.order)

    def on_market_close(self, ctx: MarketContext, result: RaceResult) -> None:
        for sel, q in self.q.items():
            self.market_log.append({
                "market_id": ctx.market_id,
                "selection_id": sel,
                "won": sel == result.winner_selection_id,
                "final_q": q,
                "fills": self.n_fills.get(sel, 0),
            })

    # ---------------------------------------------------------------------------
    # Price and risk calculations
    # ---------------------------------------------------------------------------

    def _fair_values(self, ctx: MarketContext) -> dict[int, float]:
        """Mid implied probability per runner that has both sides of the book."""
        fair: dict[int, float] = {}
        for r in ctx.runners:
            if r.removed or not r.atb or not r.atl:
                continue
            ask_p, bid_p = 1.0 / r.atb[0][0], 1.0 / r.atl[0][0]
            fair[r.selection_id] = 0.5 * (bid_p + ask_p)
        return fair

    def _variance(self, S: float, ttj: float) -> float:
        """V in price units: price risk until we exit at T_e."""
        sigma2_p = (S * (1 - S)) ** 2 * _interp(self.sigma2_x, ttj)
        return sigma2_p * max(ttj - self.greenup_seconds, 0.0)

    def _quote_odds(self, r: RunnerState, S: float, bid_p: float, ask_p: float,
                    charge_back: float, charge_lay: float) -> tuple[Optional[float], Optional[float]]:
        """(back odds, lay odds): the A-S bid and ask, widened by each side's Glosten-Milgrom charge (log-odds)."""
        q = self.q.get(r.selection_id, 0.0)
        if not (self.adverse_keep_reducing and q < 0):
            bid_p -= charge_back * S * (1 - S)    # log-odds -> price units
        if not (self.adverse_keep_reducing and q > 0):
            ask_p += charge_lay * S * (1 - S)
        return self._to_odds(r, bid_p, ask_p)

    def _as_prices(self, r: RunnerState, S: float, ttj: float) -> tuple[float, float]:
        """Avellaneda-Stoikov bid and ask (as probabilities), before any adverse-selection charge."""
        g = self.gamma
        q = self.q.get(r.selection_id, 0.0)
        V = self._variance(S, ttj)
        k_back_p = _interp(self.k_back_x, ttj) / (S * (1 - S))
        k_lay_p = _interp(self.k_lay_x, ttj) / (S * (1 - S))

        # Avellaneda-Stoikov reservation price and half-spreads
        reservation = S - q * g * V
        gc = g * self.quote_units
        ask_p = reservation + 0.5 * gc * V + math.log1p(gc / k_lay_p) / gc
        bid_p = reservation - 0.5 * gc * V - math.log1p(gc / k_back_p) / gc
        return bid_p, ask_p

    def _adverse_charges(self, quotable: list[tuple[RunnerState, float]], plain: dict[int, tuple[float, float]],
                         fair: dict[int, float], ttj: float) -> dict[int, tuple[float, float]]:
        """Glosten-Milgrom charge (log-odds) for each runner's back and lay quote."""
        if self._model is None:   # the time-binned table: the same charge on both sides
            return {r.selection_id: (charge, charge) for r, S in quotable
                    for charge in [self.adverse_scale * float(_lookup_adverse(self.adverse_x, ttj, 1.0 / S))]}

        favourite_S = max(fair.values(), default=None)
        rows, slots = [], []
        for r, S in quotable:
            f = self._features.runner_features(r, ttj, S, favourite=(S == favourite_S))
            # The model needs our quote's position, but the charge itself moves the quote, so we describe the
            # A-S quote before any charge, at the price it would be posted at
            # (so with `dime`, no more than one tick in front of the best).
            bid_odds, ask_odds = self._to_odds(r, *plain[r.selection_id])
            for side, odds in ((Side.BACK, bid_odds), (Side.LAY, ask_odds)):
                if odds is not None:
                    rows.append(AdverseFeatures.quote_row(f, side, odds))
                    slots.append((r.selection_id, side))

        charges = {r.selection_id: [0.0, 0.0] for r, _ in quotable}
        if rows:
            predicted = self._model.predict(AdverseFeatures.model_inputs(rows), num_threads=1)
            predicted = np.clip(predicted, 0.0, self.adverse_cap)
            for (sel, side), a in zip(slots, predicted):
                charges[sel][0 if side is Side.BACK else 1] = self.adverse_scale * float(a)
        return charges

    def _to_odds(self, r: RunnerState, bid_p: float, ask_p: float) -> tuple[Optional[float], Optional[float]]:
        """Bid and ask probabilities -> (back odds, lay odds) on the ladder and not crossing. None = don't quote.
        With `dime`, neither quote goes more than one tick in front of the best price on its side."""
        best_back_odds = r.atb[0][0]   # best odds anyone can back at right now (the best layer's price)
        best_lay_odds = r.atl[0][0]    # best odds anyone can lay at right now (the best backer's price)

        # Our bid = we BACK. Lower probability = higher odds, so round the odds UP.
        # A resting back waits with the backers, where lower odds is further in front.
        bid_odds = odds_round_up(1.0 / bid_p) if 0 < bid_p < 1 else None
        if bid_odds is not None and self.dime:
            front = odds_tick(best_lay_odds, -1)          # one tick in front of the best backer
            if front is not None and bid_odds < front - _EPS:
                bid_odds = front
        if bid_odds is not None and bid_odds <= best_back_odds + _EPS:
            bid_odds = odds_tick(best_back_odds, +1)      # one tick short of crossing: still resting

        # Our ask = we LAY. Higher probability = lower odds, so round the odds DOWN.
        # A resting lay waits with the layers, where higher odds is further in front.
        ask_odds = odds_round_down(1.0 / ask_p) if 0 < ask_p < 1 else None
        if ask_odds is not None and self.dime:
            front = odds_tick(best_back_odds, +1)         # one tick in front of the best layer
            if front is not None and ask_odds > front + _EPS:
                ask_odds = front
        if ask_odds is not None and ask_odds >= best_lay_odds - _EPS:
            ask_odds = odds_tick(best_lay_odds, -1)

        return bid_odds, ask_odds

    # ---------------------------------------------------------------------------
    # Order management
    # ---------------------------------------------------------------------------

    def _requote(self, ctx: MarketContext, sel: int, side: Side,
                 odds: Optional[float], units: float) -> list[Order | Cancel]:
        """
        Keep one resting quote per (runner, side) at `odds`, with at most `units` of payout
        live on that side. Orders we've cancelled still count until the cancel is confirmed,
        because with latency they can still fill - otherwise q can overshoot q_max.
        """
        current = self._quotes.get((sel, side))
        if odds is None or units <= _EPS:
            return self._cancel_slot(sel, side)
        live = self._live_units(sel, side)   # includes `current` and any pending cancels
        if current is not None and _ticks_apart(current.price, odds) == 0:
            if live <= units + 1e-6:
                return []   # already resting at the right price: keep our queue position
            return self._cancel_slot(sel, side)   # would breach the cap if it all filled

        actions = self._cancel_slot(sel, side)
        stake = round((units - live) / odds, 2)  # `current` stays live until its cancel lands
        if stake <= 0:
            return actions   # post the new quote once the old one is confirmed gone
        order = Order(ctx.market_id, sel, side, price=odds, stake=stake,
                      keep_unmatched=True, tag="quote")
        self._quotes[(sel, side)] = order
        self._open[id(order)] = [order, stake]
        actions.append(order)
        return actions

    def _live_units(self, sel: int, side: Side) -> float:
        """Payout units still able to fill on this runner and side (resting + pending cancel)."""
        return sum(rem * o.price for o, rem in self._open.values()
                   if o.selection_id == sel and o.side is side and o.tag == "quote")

    def _cancel_slot(self, sel: int, side: Side) -> list[Cancel]:
        order = self._quotes.pop((sel, side), None)
        if order is None or order.order_id is None or id(order) not in self._open:
            return []
        return [Cancel(order.order_id)]

    def _cancel_runner(self, sel: int) -> list[Cancel]:
        return self._cancel_slot(sel, Side.BACK) + self._cancel_slot(sel, Side.LAY)

    def _cancel_all(self) -> list[Cancel]:
        actions: list[Cancel] = []
        for sel, side in list(self._quotes):
            actions += self._cancel_slot(sel, side)
        return actions

    def _forget(self, order: Order) -> None:
        self._open.pop(id(order), None)
        slot = (order.selection_id, order.side)
        if self._quotes.get(slot) is order:
            del self._quotes[slot]

    def _green_up(self, ctx: MarketContext) -> list[Order]:
        """Take liquidity to flatten q on every runner (fill-or-kill, retried each snapshot)."""
        ticks = self.greenup_slippage_ticks
        pending = {o.selection_id for o, _ in self._open.values() if o.tag == "greenup"}
        orders: list[Order] = []
        for r in ctx.runners:
            q = self.q.get(r.selection_id, 0.0)
            if r.removed or r.selection_id in pending or abs(q) < self.greenup_min_units:
                continue
            if q > 0 and r.atl:
                # Long: LAY into the backers' money. Worse for a layer = higher odds.
                best = r.atl[0][0]
                side, limit = Side.LAY, odds_tick(best, +ticks) or best
            elif q < 0 and r.atb:
                # Short: BACK into the layers' money. Worse for a backer = lower odds.
                best = r.atb[0][0]
                side, limit = Side.BACK, odds_tick(best, -ticks) or best
            else:
                continue
            stake = round(abs(q) / best, 2)
            if stake <= 0:
                continue
            order = Order(ctx.market_id, r.selection_id, side, price=limit, stake=stake,
                          keep_unmatched=False, tag="greenup")
            self._open[id(order)] = [order, stake]
            orders.append(order)
        return orders