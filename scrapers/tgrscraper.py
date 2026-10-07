"""
scrapers/tgrscraper.py
---------------------

Fetches data from The Greyhound Recorder website.

I don't recommend using The Greyhound Recorder. For faster fetching and better data, use `scrapers/wdscraper.py`.
"""
import pandas as pd
import numpy as np

import datetime

import requests
import cloudscraper
from io import StringIO

def fetch_grh_data(runner_name: str, race_date: datetime.date = None,
                   drop_dnf: bool = True,
                   warn_samples: int = 0) -> pd.DataFrame:
    """Fetches Race Results data from The Greyhound Recorder for a single greyhound.

    Args:
        runner_name (str): Name of the greyhound (e.g. `"Canya Molly"`)
        race_date (datetime.date, optional): Fetch data strictly before this date (default = Today). Defaults to None.
        drop_dnf (bool, optional): Drop rows where the dog did not finish (DNF) . Defaults to True.
        warn_samples (int, optional): Warns when table fetched has less than this amount of rows. Defaults to 0.

    Returns:
        pd.DataFrame: Data for the greyhound for each race with track information, results, etc.
    """    
    if race_date is None:
        race_date = datetime.date.today()

    # Fetch data from The Greyhound Recorder
    formatted_name = runner_name.lower().replace(' ', '-').replace("'", "")
    url = f'https://www.thegreyhoundrecorder.com.au/greyhounds/{formatted_name}/'

    try:
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, timeout=3)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
    except Exception as e:
        print(f"WARNING: failed to fetch data for {runner_name}! ({e})")
        return pd.DataFrame(None)

    # Extract table
    try:
        for table in tables:
            # Race Results table
            if 'Time' in table.columns:
                table['Date'] = pd.to_datetime(table['Date'], format="%d/%m/%y")
                table = table[table['Date'] < pd.Timestamp(race_date)]
                if drop_dnf:
                    table = table[table['Fin'] != 'DNF']
                if len(table) < warn_samples:
                    print(f"WARNING: less than {warn_samples} found for {runner_name}!")
                return table
    except:
        pass
    print(f"WARNING: no race data found for {runner_name}!")
    return pd.DataFrame(None)

def fetch_grh_summary(runner_name: str) -> dict:
    """Fetches specific dog's Racing Record from The Greyhound Recorder.

    Args:
        runner_name (str): Name of the greyhound (e.g. `"Canya Molly"`).

    Returns:
        dict: Racing Record values (includes Starts, Wins, etc.).
    """    
    # Fetch data from The Greyhound Recorder
    formatted_name = runner_name.lower().replace(' ', '-').replace("'", "")
    url = f'https://www.thegreyhoundrecorder.com.au/greyhounds/{formatted_name}/'

    try:
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, timeout=3)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
    except Exception as e:
        print(f"WARNING: failed to fetch data for {runner_name}! ({e})")
        return pd.DataFrame(None)

    # Extract table
    try:
        for table in tables:
            # Racing Record table
            if 'Starts' in table.columns:
                return dict(table.iloc[0])
    except:
        pass
    print(f"WARNING: no race data found for {runner_name}!")
    return pd.DataFrame(None)
