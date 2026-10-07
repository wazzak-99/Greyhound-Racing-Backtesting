## Setup

Install the packages (from the project root):

```
pip install -r requirements.txt
```


## Market Stream Data

I didn't include market stream data since the files were too large.

You can download them from [here](https://drive.google.com/drive/folders/1qu7n6HQYACaC7Fc6dmO3GgI-saO0jrqL). Add them to `data/raw/market_stream`.

### Process the stream files

1. Put the raw files at `data/raw/market_stream/...`. Unzip the Google Drive download with its folders intact.

2. Run `python scripts/build_dataset.py` from the project root.
This writes one `.parquet` file per market to `data/processed/market_stream/...` and `race_no_lookup.json`.

(We need `race_no_lookup.json` since some race details only appear in WIN market names, so the lookup copies it across to that race's PLACE and MATCH_BET markets).

Tips:

- Test on a single day first by setting `PATTERN = "Oct/1/*/*.bz2"` at the top of `build_dataset.py`. The script prints how long it has left, so you can estimate the full month from there.
- It uses **every** CPU core by default. Lower `WORKERS` if your computer is too slow to use while it runs.
- If the run stops part-way, just run it again. Markets that already have a file are skipped.

### Columns

Each processed file has one row per runner per update. The order book is stored as `back_price_1`, `back_size_1`, `lay_price_1`, `lay_size_1` (best price), then `_2`, `_3` for the next levels. Change `LADDER_DEPTH` in `streamloaders.py` to keep more levels.

All sizes and matched amounts are converted from GBP to AUD.


## Backtesting Engine

Run a strategy with `python scripts/run_backtest.py`. Results are saved to `research/results/`.

Limitations and things that need to be implemented, none of these are a limitation of the data itself, we can implement them quickly:

1. The current engine is taker-only (reactive, fill-or-kill); proactive strategies can't be simulated right now.

2. A Strategy cannot read from multiple markets simultaneously (not implemented).

3. Strategies can't track inventory yet (not implemented).

4. The processed files keep multiple depth levels (from `streamloaders.py`). `datasources.py` now passes this to Strategy, but the execution model
still matches every order against the best level only.

5. Liquidity is never used up. Several orders in the same tick, or on consecutive ticks, can each take the full size at the best price.

6. Traded volume per price not passed to Strategy (the Strategy only knows about the general traded volume and last traded price).

7. Engine assumes zero latency.

8. The bankroll never updates after the start, only a fixed staking model is implemented.


## Other stuff

The market visualiser `data/marketvisualiser.py` only works for Windows. Run it with `python data/marketvisualiser.py path/to/market.parquet`.