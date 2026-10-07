"""
data/loaders.py
---------------
Load, clean, and index Betfair market-summary CSV files.

Not suitable for tick-by-tick signal generation. This module
is only for loading the summary layer (backtesting results, etc.)
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUMMARY_DTYPES: dict[str, str] = {
    "MARKET_ID": "int64",
    "STATE_CODE": "category",
    "TRACK": "category",
    "RACE_NO": "Int16",
    "MARKET_NAME": "string",
    "SELECTION_ID": "int64",
    "SELECTION_NAME": "string",
    "RACING_TYPE": "category",
    "RESULT": "category",
    "BSP": "float64",
    "BSP_VOLUME": "float64",
    "PREPLAY_VOLUME": "float64",
}

PRICE_COLS = [
    "BSP",
    "PREPLAY_MAX_PRICE_TAKEN",
    "PREPLAY_MIN_PRICE_TAKEN",
    "PREPLAY_LAST_PRICE_TAKEN",
    "PREPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN",
]

VOLUME_COLS = [
    "PREPLAY_VOLUME",
    "BSP_VOLUME",
]

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_summary(
    path: str | Path,
    drop_removed: bool = True,
    drop_inf_bsp: bool = True,
) -> pd.DataFrame:
    """Load a single market-summary CSV into a clean DataFrame.

    Args:
        path (str | Path): Path to the CSV file.
        drop_removed (bool, optional): Drop rows with `RESULT == 'REMOVED'` (non-runners). Defaults to True.
        drop_inf_bsp (bool, optional): Drop rows where BSP is infinite (no BSP established). Defaults to True.

    Returns:
        pd.DataFrame: \
        Added columns:
            - `IS_WINNER`
            - `TRAP`
            - `IMPLIED_BSP_PROB`
            - `MARKET_BSP_OVERROUND`
            - `FAIR_BSP_PROB`
    """    
    df = pd.read_csv(
        path,
        dtype=SUMMARY_DTYPES,
        parse_dates=["EVENT_DATE"]
    )
    
    if drop_removed:
        df = df[df["RESULT"] != "REMOVED"].copy()

    if drop_inf_bsp:
        df = df[np.isfinite(df["BSP"].astype(float))].copy()

    # Remove in-play columns
    df.drop(['INPLAY_MAX_PRICE_TAKEN',
             'INPLAY_MIN_PRICE_TAKEN',
             'LAST_PRICE_TAKEN',
             'INPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN',
             'INPLAY_VOLUME'], axis='columns', inplace=True)

    # Convenience columns
    df["IS_WINNER"] = df["RESULT"] == "WINNER"
    df["TRAP"] = (
        df["SELECTION_NAME"]
        .str.extract(r"^(\d+)\.")
        .astype("Int8")
    )
 
    # Implied probability from BSP (1/BSP)
    df["IMPLIED_BSP_PROB"] = 1.0 / df["BSP"]
 
    # Overround per market (sum of 1/BSP across runners)
    df["MARKET_BSP_OVERROUND"] = df.groupby("MARKET_ID")["IMPLIED_BSP_PROB"].transform("sum")
 
    # Fair (overround-stripped) BSP probability
    df["FAIR_BSP_PROB"] = df["IMPLIED_BSP_PROB"] / df["MARKET_BSP_OVERROUND"]
 
    return df.reset_index(drop=True)

def load_summary_h2h(
    path: str | Path,
    drop_removed: bool = True,
    drop_single_runner_market: bool = True,
) -> pd.DataFrame:
    """Load a single market summary CSV into a clean DataFrame. Specialised for H2H (AvB) files. 

    Args:
        path (str | Path): Path to the CSV file.
        drop_removed (bool, optional): Drop rows with `RESULT == 'REMOVED'` (non-runners). \
            Defaults to True.
        drop_single_runner_market (bool, optional): Drop rows of markets that only \
            recorded one runner (voided market). Defaults to True.

    Returns:
        pd.DataFrame: \
        Added columns:
            - `IS_WINNER`
            - `TRAP`
    """    
    df = pd.read_csv(
        path,
        dtype=SUMMARY_DTYPES,
        parse_dates=["EVENT_DATE"]
    )
    
    if drop_removed:
        df = df[df["RESULT"] != "REMOVED"].copy()
        df = df.dropna(subset=['SELECTION_NAME'])

    if drop_single_runner_market:
        col = df['MARKET_ID']
        mask = (col == col.shift(-1)) | (col == col.shift(1)) # is this row paired?
        df = df[mask]

    # Remove in-play columns
    df.drop(['INPLAY_MAX_PRICE_TAKEN',
             'INPLAY_MIN_PRICE_TAKEN',
             'LAST_PRICE_TAKEN',
             'INPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN',
             'INPLAY_VOLUME'], axis='columns', inplace=True)

    # Convenience columns
    df["IS_WINNER"] = df["RESULT"] == "WINNER"
    df["TRAP"] = (
        df["SELECTION_NAME"]
        .str.extract(r"^(\d+)\.")
        .astype("Int8")
    )
 
    return df.reset_index(drop=True)

def load_summary_glob(
    pattern: str,
    drop_removed: bool = True,
    drop_inf_bsp: bool = True
) -> pd.DataFrame:
    """Load and concatenate all CSVs matching a glob pattern.

    Examples:
        >>> df = load_summary_glob("data/raw/market_summary/*.csv)

    Raises:
        FileNotFoundError: No files matched the glob pattern.
    """    
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    frames = [load_summary(f, drop_removed, drop_inf_bsp) for f in files]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Market-level data
# ---------------------------------------------------------------------------

def to_market_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapses runner-level summary into market-level summary (one row per market). \
    Useful for market-level EDA, not for backtesting.

    Args:
        df (pd.DataFrame): Runner-level market-summary files \
            loaded from `load_summary()`

    Returns:
        pd.DataFrame: Market-level aggregated data with winner stats attached
    """    
    winner = df[df["IS_WINNER"]].set_index("MARKET_ID")[
        ["SELECTION_ID", "SELECTION_NAME", "TRAP", "BSP", "FAIR_BSP_PROB"]
    ].add_prefix("WINNER_")
 
    market = (
        df.groupby("MARKET_ID")
        .agg(
            EVENT_DATE=("EVENT_DATE", "first"),
            STATE_CODE=("STATE_CODE", "first"),
            TRACK=("TRACK", "first"),
            RACE_NO=("RACE_NO", "first"),
            MARKET_NAME=("MARKET_NAME", "first"),
            N_RUNNERS=("SELECTION_ID", "count"),
            TOTAL_BSP_VOLUME=("BSP_VOLUME", "sum"),
            TOTAL_PREPLAY_VOLUME=("PREPLAY_VOLUME", "sum"),
            OVERROUND=("MARKET_BSP_OVERROUND", "first"),
        )
        .join(winner)
        .reset_index()
    )
    return market

def to_market_level_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Collapses runner-level summary into market-level summary (one row per market). \
    Useful for market-level EDA, not for backtesting. Specialised for H2H (AvB) files.

    Args:
        df (pd.DataFrame): Runner-level market-summary files \
            loaded from `load_summary_h2h()`

    Returns:
        pd.DataFrame: Market-level aggregated data with winner stats attached
    """    
    winner = df[df["IS_WINNER"]].set_index("MARKET_ID")[
        ["SELECTION_ID", "SELECTION_NAME", "TRAP"]
    ].add_prefix("WINNER_")
 
    market = (
        df.groupby("MARKET_ID")
        .agg(
            EVENT_DATE=("EVENT_DATE", "first"),
            STATE_CODE=("STATE_CODE", "first"),
            TRACK=("TRACK", "first"),
            RACE_NO=("RACE_NO", "first"),
            MARKET_NAME=("MARKET_NAME", "first"),
            N_RUNNERS=("SELECTION_ID", "count"),
            TOTAL_PREPLAY_VOLUME=("PREPLAY_VOLUME", "sum"),
        )
        .join(winner)
        .reset_index()
    )
    return market

# ---------------------------------------------------------------------------
# Other Helper Functions
# ---------------------------------------------------------------------------
 
 
def summary_stats(df: pd.DataFrame) -> None:
    """Print a quick diagnostic summary of a loaded dataframe."""
    n_markets = df["MARKET_ID"].nunique()
    n_runners = len(df)
    date_range = f"{df['EVENT_DATE'].min().date()} → {df['EVENT_DATE'].max().date()}"
    avg_field = df.groupby("MARKET_ID").size().mean()
    avg_bsp = df.loc[df["IS_WINNER"], "BSP"].median()
 
    print(f"Markets      : {n_markets:,}")
    print(f"Runners      : {n_runners:,}")
    print(f"Date range   : {date_range}")
    print(f"Avg field    : {avg_field:.2f}")
    print(f"Median winner BSP: {avg_bsp:.2f}")
    print(f"States       : {', '.join(df['STATE_CODE'].unique().tolist())}")
