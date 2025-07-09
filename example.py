from decimal import Decimal, ROUND_05UP
import openmeteo_requests  # Open-Meteo's Python client for weather API
import pandas as pd
import requests_cache      # Used to cache API requests and avoid duplicate calls
from retry_requests import retry  # Automatically retries failed HTTP requests
import os
import json
from collections import defaultdict

input = "/Users/kateliu/Documents/GitHub/Traffic-density-pipeline/data/processed/input/traffic_data_sample.geojson"
output_dir = "/Users/kateliu/Documents/GitHub/Traffic-density-pipeline/data/processed/output/"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
output_path = os.path.join(output_dir, "traffic_data_with_weather.geojson")

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

#Loading old data
# === Load traffic observation data from GeoJSON ===
with open(input, 'r') as f:
    traffic_data = json.load(f)

features = traffic_data['features']  # Going through all the GeoJSON features (traffic points)
location_time = defaultdict(list)    # Dictionary holding coordinates:timestamps

# === Group timestamps by unique location ===
# This helps avoid duplicate weather API calls for the same spot and time range
for feature in features:
    coordinates = tuple(feature['geometry']['coordinates'])  # breaking geoson into (longitude, latitude)
    timestamp = feature['properties']['Timestamp']           # e.g., "2016-01-04 00:00:00"
    location_time[coordinates].append(timestamp)
#Loading old data


url = "https://archive-api.open-meteo.com/v1/archive"

for coordinates, timestamps in location_time.items():
    lon, lat = coordinates 
    #Arounded to 2 decimal places for better precision in API calls
    long = Decimal(lon).quantize(Decimal('0.01'), rounding=ROUND_05UP)
    lati = Decimal(lat).quantize(Decimal('0.01'), rounding=ROUND_05UP)
    start_date = min(timestamps).split("T")[0]  # Use earliest timestamp (only date part) --> idk
    end_date = max(timestamps).split("T")[0]    # Use latest timestamp --> idk

    # === Construct API URL and parameters ===
    params = {
        "latitude": lati, 
        "longitude": long,
        # "start_date": "2024-01-01",     # Static start for now — can be changed to dynamic if needed # 2016-01-01
        # "end_date": "2024-06-10",       # Static end date
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m" ,"precipitation","cloud_cover","cloud_cover_low","cloud_cover_mid","cloud_cover_high", "wind_speed_10m"],
        "timezone": "America/New_York"
    }
    
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    # The variables are accessed based on their positions on the list
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()  
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(2).ValuesAsNumpy()
    hourly_cloud_cover_low = hourly.Variables(3).ValuesAsNumpy()
    hourly_cloud_cover_mid = hourly.Variables(4).ValuesAsNumpy()
    hourly_cloud_cover_high = hourly.Variables(5).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(6).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),freq = pd.Timedelta(seconds = hourly.Interval()),inclusive = "left")}

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["percipitation"] = hourly_precipitation
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
    hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
    hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    print(hourly_dataframe)
    
    