"""
scrapers/wdscraper.py
--------------------

Module to fetch data from the [Watchdog GRV website](https://watchdog.grv.org.au/) and API.
"""
import pandas as pd
import requests
import numpy as np

import datetime
import regex as re

class race_t:
    def __init__(self, id: int, location: str, date: str, dist: int, rain: int = 0, temp: int = 0, hum: int = 0, times=None):
        self.id = id
        self.location = location
        self.date = date
        self.dist = dist
        self.rain = rain
        self.temp = temp
        self.hum = hum
        self.times = [] if times is None else times
    
    def __repr__(self):
        return f'({self.id}, {self.location}, {self.date})'
    
HEADERS = {'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}
TIMEOUT_SECONDS = 3

# ---------------------------------------------------------------------------
# Market-level data / Isolynx
# ---------------------------------------------------------------------------

def fetch_wd_api(date: datetime.date | str, warn: bool = False) -> dict | None:
    """Fetches Watchdog public API data for dog races.

    Args:
        date (datetime.date): The date to search from. \
            Watchdog seems to store all upcoming races around a month from this date.

    Returns:
        dict: Dictionary of race names, IDs, etc.

    Examples:
        >>> fetch_wd_api(datetime.date(2025, 10, 1))
        
        >>> fetch_wd_api('2025-10-01')
    """    
    formatted_date = date
    if type(date) == datetime.date:
        formatted_date = date.strftime('%Y-%m-%d')
    url = f'https://watchdog.grv.org.au/api/public/form/calendar-month/{formatted_date}T00:00:00.000Z'
    try:
        data = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS).json()
        return dict(data)
    except:
        if warn:
            print(f"WARNING: failed to fetch data at .../calendar-month/{formatted_date}...!")

def _get_race_id(meeting_id: int, race_number: int, warn: bool = False) -> int | None: 
    """Gets the race ID for Isolynx, given the `meeting_id` and `race_number`.

    Args:
        meeting_id (int): The meeting ID.
        race_number (int): The race number.
        warn (bool, optional): Enable warnings. Defaults to False.

    Returns:
        int | None: the race ID.
    """    
    url = f'https://watchdog.grv.org.au/api/public/form/meeting/{meeting_id}'
    try:
        data = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS).json()
        if len(data['races']) == 0 and warn:
            print(f"WARNING: data requested at .../meeting/{meeting_id} is empty!")

        for race in data['races']:
            if(race['raceNumber'] == race_number):
                return race['id']
    except:
        if warn:
            print(f"WARNING: failed to fetch data at .../meeting/{meeting_id}!")

def get_race_ids(meeting_id: int, warn: bool = False) -> list[int]:
    """Gets all race IDs for Isolynx, given the `meeting_id`.

    Args:
        meeting_id (int): The meeting ID.
        warn (bool, optional): Enable warnings. Defaults to False.

    Returns:
        list[int]: A list of all race IDs.
    """    
    url = f'https://watchdog.grv.org.au/api/public/form/meeting/{meeting_id}'
    race_ids = []
    try:
        data = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS).json()
        if len(data['races']) == 0 and warn:
            print(f"WARNING: data requested at .../meeting/{meeting_id} is empty!")

        for race in data['races']:
            if race["id"]: race_ids.append(race["id"])
    except:
        if warn:
            print(f"WARNING: failed to fetch data at .../meeting/{meeting_id}!")
    
    return race_ids

def get_races(meeting_id: int, warn: bool = False):
    """
    Gets all races for a given meeting.

    Returns a list of all these races (race IDs, date-time, and location).
    """
    url = f'https://watchdog.grv.org.au/api/public/form/meeting/{meeting_id}'
    races = []
    try:
        data = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS).json()
        if len(data['races']) == 0 and warn:
            print(f"WARNING: data requested at .../meeting/{meeting_id} is empty!")

        for race in data['races']:
            if race["id"]:
                start_time = race["startTime"]
                start_time = start_time[:10] + " " + start_time[11:]
                race_object = race_t(race['id'], data["meetings"][0]["trackName"], start_time, race["distance"])
                races.append(race_object)

    except:
        if warn:
            print(f"WARNING: failed to fetch data at .../meeting/{meeting_id}!")
    
    return races

def fetch_wd_isolynx(meeting_id: int, race_number: int, warn: bool = False) -> dict | None:
    """Fetches Isolynx splits data from the Watchdog public API.

    Args:
        meeting_id (int): The meeting ID.
        race_number (int): The race number.
        warn (bool, optional): Enable warnings. Defaults to False.

    Returns:
        dict | None: Dictionary of Isolynx splits data.
    """    
    race_id = _get_race_id(meeting_id, race_number)
    url = f'https://watchdog.grv.org.au/api/public/isolynx/{race_id}/splits'
    try:    
        data = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS).json()
        if len(data['splits']) == 0 and warn:
            print(f"WARNING: data requested at .../isolynx/{race_id}/splits is empty!")

        return dict(data)
    except:
        if warn:
            print(f"WARNING: failed to fetch data at .../isolynx/{race_id}/splits!")


# ---------------------------------------------------------------------------
# Dog-level data 
# ---------------------------------------------------------------------------

def _norm_dog_name(s):
    return ' '.join(re.sub(r'[^\w\s]', '', s.lower()).split())

def get_dog_id(search_term: str, exact_match: bool = True, take: int = 1) -> int | dict | None:
    """Searches Watchdog API with `search_term` and returns the dog ID if found.

    Args:
        search_term (str): The name of the dog to search for (case-insensitive).
        exact_match (bool, optional): Only return if an exact match is found. Defaults to True.
        take (int, optional): How many dogs to search for. Defaults to 5.

    Returns:
        int | dict: Returns a dictionary of dog names as keys and their dog IDs as values. \
        If `exact_match` is `True`, the single dog ID as an integer is returned.

    Examples:
        >>> get_dog_id('Canya Molly')
        965774414
    """    
    params = {'skip': 0, 'take': take, 'reverse': 0, 'q': search_term, 'mode': 'all'}

    try:
        response = requests.get('https://watchdog.grv.org.au/api/public/search', 
                                params=params, headers=HEADERS, timeout=TIMEOUT_SECONDS
                                ).json()

        data = {}
        for entry in response['entries']:
            data[entry['name']] = entry['id']

        if exact_match:
            for name in data:
                if _norm_dog_name(search_term) == _norm_dog_name(name):
                    return data[name]
            else:
                print(f"WARNING: Exact search failed when trying to get dog id, search_term: {search_term}!")
        else:
            return data
    except:
        print(f"WARNING: failed to get dog_id, search_term: {search_term}!")
    
def fetch_dog_summary(dog_id: int) -> dict | None:
    """Fetches summary data for a specific dog from Watchdog API.

    Args:
        dog_id (int): The Dog ID. Get from `get_dog_id()`.

    Returns:
        dict: \
        Includes: `name`, `sex`, `last5`, `last_race_date`, `career_starts`, `career_wins` \
        `career_2nd`, `career_3rd`, `prize_money`.

    Examples:
        >>> wd.fetch_dog_summary(965774414)
        {
        'name': 'Canya Molly',
        'sex': 'Bitch', 
        'last5': '43543', 
        'last_race_date': datetime.datetime(2026, 6, 28, 12, 14), 
        'career_starts': 48, 
        'career_wins': 12, 
        'career_2nd': 7, 
        'career_3rd': 10, 
        'prize_money': 17610
        } 
        
    ## Notes:
        - `last5` is in oldest-to-latest order of recency.
    """    
    url = f'https://watchdog.grv.org.au/api/public/form/dog/{dog_id}'
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS).json()
        dog = response['bbsubjects'][0]
        
        data = {}
        data['name'] = dog['name'] 
        data['sex'] = dog['sex']

        data['last5'] = dog['last5']
        data['last_race_date'] = datetime.datetime.fromisoformat(
            dog['lastRun']['startTime'].replace("Z", "+00:00")
            ).date()

        
        data['career_starts'] = dog['stats']['career']['starts']
        data['career_wins'] = dog['stats']['career']['first']
        data['career_2nd'] = dog['stats']['career']['second']
        data['career_3rd'] = dog['stats']['career']['third']
        data['prize_money'] = dog['careerPrizeMoney']
        
        return data
    except:
        print(f"WARNING: failed to fetch dog summary, dog_id: {dog_id}!")

def fetch_dog_form(dog_id: int) -> pd.DataFrame: 
    """Fetches form data for a specific dog from Watchdog API

    Args:
        dog_id (int): The Dog ID. Get from `get_dog_id()`.

    Returns:
        pd.DataFrame: \
        Includes: `meeting_id`, `race_id`, `race_number`, `grade`, `dist`, `count_runners`, `box`, \
        `start_time`, `track_code`, `time`, `resultPlace`, `pir`, `starting_price`.

    Examples:
        >>> fetch_dog_form(965774414)
        scraper/wd_form_canya_molly_example.csv
    """    
    url = f'https://watchdog.grv.org.au/api/public/form/dog/{dog_id}'
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS).json()
        form = response['bbsubjects'][0]['recentForm']

        data = []
        for race in form:
            race_data = {}


            race_data['meeting_id'] = race['meetingId']
            race_data['race_id'] = race['raceId']
            race_data['race_number'] = race['raceNumber']

            race_data['grade'] = race['gradeCode']
            race_data['dist'] = race['distance']
            race_data['count_runners'] = race['countDogsStarted']
            race_data['box'] = race['box']
            race_data['start_time'] = datetime.datetime.fromisoformat(
                race['startTime'].replace("Z", "+00:00")
                ).date()
            race_data['track_code'] = race['trackCode']

            race_data['time'] = race['resultTime']
            race_data['place'] = race['resultPlace']
            if race['pir'] == None or race['pir'] == '0':
                race_data['pir'] = None

            race_data['starting_price'] = race['startingPrice']

            data.append(race_data)

        return pd.DataFrame(data)
    except:
        print(f"WARNING: failed to fetch dog form, dog_id: {dog_id}!")
        
