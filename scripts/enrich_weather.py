from decimal import Decimal, ROUND_05UP
import openmeteo_requests  # Open-Meteo's Python client for weather API
import pandas as pd
import requests_cache  # Used to cache API requests and avoid duplicate calls
from retry_requests import retry  # Automatically retries failed HTTP requests
import os
import json
from collections import defaultdict
from datetime import datetime
import time
from openmeteo_requests.Client import OpenMeteoRequestsError


# === Setup input/output paths ===
# Get the base directory of the project (go up one level from /scripts)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_file = os.path.join(BASE_DIR, "data", "processed", "traffic_data.geojson")
output_dir = os.path.join(BASE_DIR, "data", "processed", "output")
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
cache_path = os.path.join(BASE_DIR, "data", "cache", "weather_cache")
os.makedirs(os.path.dirname(cache_path), exist_ok=True)
progress_log_path = os.path.join(output_dir, "weather_processing_progress.txt")

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession(cache_path, expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

request_count_hour = 0
request_count_day = 0
hour_start = time.time()
day_start = time.time()


# Loading old data
# === Load traffic observation data from GeoJSON ===
with open(input_file, 'r') as f:
    traffic_data = json.load(f)

features = traffic_data['features']  # Going through all the GeoJSON features (traffic points)
timestamps = [feature['properties']['Timestamp'] for feature in traffic_data['features']] # go through each timestamp in the input file and store it in a list
start_date = min(datetime.fromisoformat(ts).date() for ts in timestamps) # for every timestamp in the list, get the max and store it in var
end_date = max(datetime.fromisoformat(ts).date() for ts in timestamps)
print(f"Latest timestamp: {end_date}")
print(f"Earliest timestamp: {start_date}")
location_time = defaultdict(list)  # Dictionary holding coordinates:timestamps
processed_count = 0
processed_features = 0

# === Group timestamps by unique location ===
# This helps avoid duplicate weather API calls for the same spot and time range
for feature in features:
    coordinates = tuple(feature['geometry']['coordinates'])  # breaking geoson into (longitude, latitude)
    timestamp = feature['properties']['Timestamp']  # e.g., "2016-01-04 00:00:00"
    location_time[coordinates].append(timestamp)
    processed_features+=1
    with open(progress_log_path, "a") as log:
        log.write(f"Processed features: {processed_features}/{len(features)}\n") # log to keep track of what's been processed

url = "https://archive-api.open-meteo.com/v1/archive"

all_weather_data = []

for coordinates, timestamps in location_time.items():
    lon, lat = coordinates
    # Arounded to 2 decimal places for better precision in API calls
    long = Decimal(lon).quantize(Decimal('0.01'), rounding=ROUND_05UP)
    lati = Decimal(lat).quantize(Decimal('0.01'), rounding=ROUND_05UP)
    #start_date = min(timestamps).split("T")[0]  # Use earliest timestamp (only date part) --> idk
    #end_date = max(timestamps).split("T")[0]  # Use latest timestamp --> idk

    # === Construct API URL and parameters ===
    params = {
        "latitude": lati,
        "longitude": long,
        # "start_date": "2010-01-01",     # Static start for now — can be changed to dynamic if needed # 2016-01-01
        # "end_date": "2024-06-10",       # Static end date
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "precipitation", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "cloud_cover", "wind_speed_10m", "snow_depth", "visibility", "apparent_temperature", "relative_humidity_2m", "weather_code", "freezing_level_height", "uv_index"],
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/New_York"
    }

    while True:
        # Check elapsed time since hour/day start
        now = time.time()
        elapsed_hour = now - hour_start
        elapsed_day = now - day_start

        # Reset counters if needed
        if elapsed_hour >= 3600:
            request_count_hour = 0
            hour_start = now
        if elapsed_day >= 86400:
            request_count_day = 0
            day_start = now

        # Throttle based on safe limits
        if request_count_hour >= 4500:
            sleep_time = 3600 - elapsed_hour
            print(f"[Rate limit] Hourly threshold near. Sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
            continue
        if request_count_day >= 9500:
            sleep_time = 86400 - elapsed_day
            print(f"[Rate limit] Daily threshold near. Sleeping {sleep_time / 3600:.2f} hours")
            time.sleep(sleep_time)
            continue

        try:
            responses = openmeteo.weather_api(url, params=params)
            request_count_hour += 1
            request_count_day += 1
            time.sleep(0.8)  # Space out calls to avoid bursts
            break
        except OpenMeteoRequestsError as e:
            if "Minutely API request limit exceeded" in str(e):
                print("Minutely rate limit hit. Sleeping for 10 seconds...")
                with open(progress_log_path, "a") as log:
                    log.write("Minutely rate limit hit. Sleeping for 10 seconds...\n")
                time.sleep(10)
            else:
                raise e

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
    hourly_snow_depth = hourly.Variables(7).ValuesAsNumpy()
    hourly_visibility = hourly.Variables(8).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(9).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(10).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(11).ValuesAsNumpy()
    hourly_freezing_level_height = hourly.Variables(12).ValuesAsNumpy()
    hourly_uv_index = hourly.Variables(13).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "latitude": float(lat),
        "longitude": float(lon),
        "temperature_2m": hourly_temperature_2m, # Extreme heat or cold can affect vehicle performance and road safety.
        "precipitation": hourly_precipitation,
        "cloud_cover": hourly_cloud_cover, # Low visibility and poor lighting conditions impact driving.
        "cloud_cover_low": hourly_cloud_cover_low,
        "cloud_cover_mid": hourly_cloud_cover_mid,
        "cloud_cover_high": hourly_cloud_cover_high,
        "wind_speed_10m": hourly_wind_speed_10m, # Sudden gusts can be dangerous for vehicles, especially trucks
        "snow_depth": hourly_snow_depth, #  Impacts vehicle traction and congestion
        "visibility": hourly_visibility, # Critical for driving safety, especially in fog, rain, or snow
        "apparent_temperature": hourly_apparent_temperature, # Perceived temperature may correlate better with driver behavior than raw temp.
        "relative_humidity_2m": hourly_relative_humidity_2m, # High humidity can mean foggy conditions or slippery roads
        "weather_code": hourly_weather_code, # Categorical description of weather (e.g., clear, fog, rain, storm, etc.)
        "freezing_level_height": hourly_freezing_level_height, # Can help model icy road conditions in winter
        "uv_index": hourly_uv_index # sun glare can impair vision, though not as important
    }

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    print(hourly_dataframe)
    all_weather_data.append(hourly_dataframe)

    # Save cumulative progress after each successful response
    partial_csv_path = os.path.join(output_dir, "weather_data_partial.csv")
    pd.concat(all_weather_data, ignore_index=True).to_csv(partial_csv_path, index=False)

    processed_count+=1
    with open(progress_log_path, "a") as log:
        log.write(f"Processed coords/times: {processed_count}/{len(location_time.items())}\n")

# Combine all location DataFrames into one
weather_df = pd.concat(all_weather_data, ignore_index=True)
# Save to CSV
weather_df.to_csv(os.path.join(output_dir, "weather_data.csv"), index=False)
