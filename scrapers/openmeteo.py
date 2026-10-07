"""
scrapers/openmeteo.py
---------------------

Fetches weather data (single or multiple locations) from Open-Meteo's API. 
The data is cached to avoid repeated requests for the same location and time period.
"""
import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

def get_weather_data(start_date: str = "2025-01-01", end_date: str = "2025-12-31"):
	location_search = pd.read_csv('location_search.csv')
	print(location_search.columns.tolist())
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	url = "https://archive-api.open-meteo.com/v1/archive"
	params = {
		"latitude": location_search['LAT'].tolist(),
		"longitude": location_search['LONG'].tolist(),
		"start_date": start_date,
		"end_date": end_date,
		"hourly": ["temperature_2m", "relative_humidity_2m", "rain"],
	}
	responses = openmeteo.weather_api(url, params = params)

	return responses, location_search

def get_weather(date: str, location: str, responses, parameter_index: int, location_search) -> float:
	"""Returns the 5-hour rolling average weather at a particular location and date-time.

	Args:
		date (str): A string starting with the format `'YYYY-MM-DD HH'`.
		location (str): The meeting's track name (sourced from Watchdog's API). 
		responses : Weather data for all locations for the entirety of a time period. \
			Use `get_weather_data()` to fetch this.
		parameter_index (int): \
			- `1`: Temperature
			- `2`: Humidity
			- `3`: Rain
		location_search (_type_): _description_		
	
	Returns:
		float: _description_
	"""	
	# Process this location's data
	response = responses[location_search.index[location_search['TRACK'] == location].tolist()[0]]

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
	hourly_rain = hourly.Variables(2).ValuesAsNumpy()

	hourly_data = {
		"date": pd.date_range(
			start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
			end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
			freq = pd.Timedelta(seconds = hourly.Interval()),
			inclusive = "left"
		)
	}

	hourly_data["temperature_2m"] = hourly_temperature_2m
	hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
	hourly_data["rain"] = hourly_rain

	hourly_dataframe = pd.DataFrame(data = hourly_data)
	mask = hourly_dataframe['date'].astype(str).str[:13] == date[:13]
	race_index = hourly_dataframe.index[mask].tolist()[0]
	return hourly_dataframe.iloc[race_index-4:race_index+1, parameter_index].mean()