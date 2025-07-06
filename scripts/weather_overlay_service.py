import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import os
import json
from collections import defaultdict

# TODO: Load your point-based traffic GeoJSON file containing coordinates and timestamps
input = "C:/Users/nokon/SWE/AI4AllProj/data/processed/input/traffic_data_sample.geojson"
output_dir = "C:/Users/nokon/SWE/AI4AllProj/data/processed/output/"
os.makedirs(output_dir, exist_ok=True)

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# loading in the traffic data as a dictionary
with open(input, 'r') as f:
    traffic_data = json.load(f)

features = traffic_data['features']
location_time = defaultdict(list) # data required for making the api call

# loop through each point(feature) and request weather data
for feature in features:
    coordinates = tuple(feature['geometry']['coordinates'])
    properties = feature['properties']
    request_id = properties['RequestID']
    timestamp = properties['Timestamp']
    key = (coordinates, request_id) # ensuring no duplicate requests for 1 area
    location_time[key].append(timestamp)
    
    

# Match each timestamped traffic point with its corresponding weather dataTODO: For each feature in that file:
# 
for (coordinates, request_id), timestamps in location_time.items():
    lon, lat = coordinates
    start_date = min(timestamps).split("T")[0]  # Extract date part from ISO timestamp
    end_date = max(timestamps).split("T")[0]

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2016-01-01",
        "end_date": "2024-06-10",
        "hourly": "temperature_2m",
        "timezone": "America/New_York",
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch", 
        "cloud_cover":"%", 
        "cloud_cover_low":"%",
        "cloud_cover_med":"%",
        "cloud_cover_high":"%"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]  # Assuming we only need the first response for this point



#     - Extract longitude, latitude from 'geometry'['coordinates']
#     - Extract timestamp from 'properties'['Timestamp']
#     - Format timestamp into the required ISO format for Open-Meteo API
#     - Construct Open-Meteo Historical Weather API URL with parameters:
#         * latitude
#         * longitude
#         * start_date and end_date (same day as timestamp)
#         * hourly variables (e.g., temperature_2m, precipitation, etc.)
#     - Make the API request
#     - Parse the returned weather data (temperature, precipitation, etc.)
#     - Attach weather data to the feature's 'properties' under a 'Weather' key

# TODO: Save the enriched GeoJSON with weather overlays for debugging/inspection if needed

# TODO: Modify the FastAPI `/map` endpoint:
#     - Overlay traffic lines as before (already implemented)
#     - Overlay points with weather data:
#         * Use folium.CircleMarker or folium.Marker at each point
#         * Use color, radius, or popup text to represent temperature or precipitation on that timestamp
#         * Example: popup showing "Temp: XX°C, Rain: YYmm, Date: YYYY-MM-DD HH:MM"

# TODO: Ensure all overlays are added to the same folium.Map instance so weather overlays sit atop traffic lines

# TODO: Save the final map as HTML and return using FileResponse as before
