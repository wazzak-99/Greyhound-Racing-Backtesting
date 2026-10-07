## Market Stream Data

I didn't include market stream data since the files were too large.

You can download them from [here](https://drive.google.com/drive/folders/1qu7n6HQYACaC7Fc6dmO3GgI-saO0jrqL). Add them to `data/raw/market_stream`.

### Process the stream files

1. Put the raw files at `data/raw/market_stream/...`. Unzip the Google Drive download with its folders intact.

2. Run `python scripts/build_dataset.py` from the project root.
This writes one CSV per market to `data/processed/market_stream/...` and `race_no_lookup.json`. 

(We need `race_no_lookup.json` since some race details only appear in WIN market names, so the lookup copies it across to that race's PLACE and MATCH_BET markets).


## Backtesting Engine

Limitations and things that need to be implemented:

1. The current engine is taker-only (reactive, fill-or-kill); proactive strategies can't be simulated right now. 

2. Strategies can't track inventory yet (not implemented).

3. The CSVs maintain 3 levels (from `streamloaders.py`), but `datasources.py` only passes the best BACK and LAY prices to the Strategy.

4. Traded volume per price not passed to Strategy (the Strategy only knows about the general traded volume and last traded price)

5. Engine assumes zero latency.


## Other stuff

The market visualiser `data/marketvisualiser.py` only works for Windows.

