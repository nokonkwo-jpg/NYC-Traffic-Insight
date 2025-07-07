import openmeteo_requests  # Open-Meteo's Python client for weather API
import pandas as pd
import requests_cache      # Used to cache API requests and avoid duplicate calls
from retry_requests import retry  # Automatically retries failed HTTP requests
import os
import json
from collections import defaultdict
from datetime import datetime

# === Setup input/output paths ===
input = "/Users/kateliu/Documents/GitHub/Traffic-density-pipeline/data/processed/input/traffic_data_sample.geojson"
output_dir = "/Users/kateliu/Documents/GitHub/Traffic-density-pipeline/data/processed/output/"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
output_path = os.path.join(output_dir, "traffic_data_with_weather.geojson")

# === Set up cached and retryable HTTP session for Open-Meteo API ===
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)  # Cache expires in 1 hour
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)  # Retry up to 5 times with delay
openmeteo = openmeteo_requests.Client(session=retry_session)  # Create Open-Meteo client

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

# === Query weather data from Open-Meteo for each unique location ===
weather_lookup = {}  # Dictionary that stores weather data for each location+timestamp

for coordinates, timestamps in location_time.items():
    lon, lat = coordinates
    start_date = min(timestamps).split("T")[0]  # Use earliest timestamp (only date part) --> idk
    end_date = max(timestamps).split("T")[0]    # Use latest timestamp --> idk

    # === Construct API URL and parameters ===
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-01-01",     # Static start for now — can be changed to dynamic if needed # 2016-01-01
        "end_date": "2024-06-10",       # Static end date
        "hourly": ["temperature_2m" ,"precipitation","cloud_cover","cloud_cover_low","cloud_cover_mid","cloud_cover_high", "wind_speed_10m"],
        "timezone": "America/New_York"
    }

    # === Make API request ===
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]  # Only one response is expected for each location
    print(response)
    print("Hi")
    
       # === Extract weather variables ===
    response = responses[0]
    hourly = response.Hourly()
    times = pd.to_datetime(hourly.Time(), unit='s', utc=True).strftime("%Y-%m-%dT%H:00:00")

    temperature = hourly.Variables(0).ValuesAsNumpy()
    precipitation = hourly.Variables(1).ValuesAsNumpy()
    cloud_cover = hourly.Variables(2).ValuesAsNumpy()
    cloud_cover_low = hourly.Variables(3).ValuesAsNumpy()
    cloud_cover_mid = hourly.Variables(4).ValuesAsNumpy()
    cloud_cover_high = hourly.Variables(5).ValuesAsNumpy()
    wind_speed = hourly.Variables(6).ValuesAsNumpy()


    # === Build a dictionary of timestamp → weather data
    weather_by_time = {
        t: {
            "temperature_2m": float(temperature[i]),
            "precipitation": float(precipitation[i]),
            "cloud_cover": float(cloud_cover[i]),
            "cloud_cover_low": float(cloud_cover_low[i]),
            "cloud_cover_mid": float(cloud_cover_mid[i]),
            "cloud_cover_high": float(cloud_cover_high[i]),
            "wind_speed_10m": float(wind_speed[i])
        }
        for i, t in enumerate(times)
    }

    # === Store weather data in the lookup table
    for ts in timestamps:
        # Format traffic timestamp to ISO format: 'YYYY-MM-DDTHH:00'
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
        dt_iso = dt.replace(minute=0, second=0).isoformat()
        if dt_iso in weather_by_time:
            weather_lookup[(coordinates, ts)] = weather_by_time[dt_iso]
        else:
            print(f"Warning: No weather match for {coordinates} at {dt_iso}")
    # === Attach weather data to each traffic point
    for feature in features:
        coords = tuple(feature['geometry']['coordinates'])
        ts = feature['properties']['Timestamp']
        weather = weather_lookup.get((coords, ts))
        if weather:
            feature['properties']['Weather'] = weather

    # === Save enriched GeoJSON
    with open(output_path, 'w') as f:
        json.dump(traffic_data, f, indent=2)

    print(f"✅ Enriched traffic data saved to: {output_path}")
