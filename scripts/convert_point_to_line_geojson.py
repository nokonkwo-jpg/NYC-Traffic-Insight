import json
import requests
import os
# this is a script to convert the traffic geojson file to a geojson line format for use in the map.
# Specifically for heatmap use

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INPUT_FILE = r"C:\Users\nokon\Downloads\traffic_data.geojson"
OUTPUT_FILE = r"C:\Users\nokon\Downloads\traffic_data_lines.geojson"
CACHE_FILE = "geocode_cache.json"
CHECKPOINT_FILE = "progress_checkpoint.json"
SAVE_INTERVAL = 100  # save progress every N features

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        geocode_cache = json.load(f)
else:
    geocode_cache = {}

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint = json.load(f)
    start_idx = checkpoint.get("last_index", 0)
else:
    start_idx = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

features = data["features"]
print(f"Total features to process: {len(features)}")
output_features = []

if os.path.exists(OUTPUT_FILE) and start_idx > 0:
    with open(OUTPUT_FILE, "r") as f:
        output_data = json.load(f)
        output_features = output_data["features"]

def geocode_location(location_str):
    if location_str in geocode_cache:
        return geocode_cache[location_str]
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location_str}&key={GOOGLE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get("results")
        if results:
            loc = results[0]["geometry"]["location"]
            coords = [loc["lng"], loc["lat"]]
            geocode_cache[location_str] = coords
            with open(CACHE_FILE, "w") as f:
                json.dump(geocode_cache, f)
            return coords
    print(f"Geocoding failed for: {location_str}")
    return None

for idx in range(start_idx, len(features)):
    feature = features[idx]
    props = feature["properties"]
    borough = props.get("Borough", "")
    street = props.get("Street", "")
    from_addr = props.get("From", "")
    to_addr = props.get("To", "")

    from_location = f"{from_addr}, {street}, {borough}, NYC"
    to_location = f"{to_addr}, {street}, {borough}, NYC"

    from_coords = geocode_location(from_location)
    to_coords = geocode_location(to_location)

    if from_coords and to_coords:
        new_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [from_coords, to_coords]
            },
            "properties": props
        }
        output_features.append(new_feature)

    if (idx + 1) % SAVE_INTERVAL == 0 or idx == len(features) - 1:
        # Save progress
        with open(OUTPUT_FILE, "w") as f:
            json.dump({
                "type": "FeatureCollection",
                "features": output_features
            }, f)
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"last_index": idx + 1}, f)
        print(f"Saved progress at feature {idx + 1}/{len(features)}")

print("Conversion complete. Lines saved to", OUTPUT_FILE)
