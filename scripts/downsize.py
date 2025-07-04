import json
# this is a script to downsize the geojson file to a smaller size for testing purposes.

input_file = r"C:\Users\nokon\SWE\AI4AllProj\data\processed\traffic_data.geojson" # change this path to data/traffic_data.geojson
output_file = r"C:\Users\nokon\SWE\AI4AllProj\data\processed\traffic_data_small_50.geojson" # same with this one
N = 200  # number of features to keep for testing

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

small_geojson = {
    "type": "FeatureCollection",
    "features": data["features"][:N]
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(small_geojson, f)

print(f"Downsampled {N} features written to {output_file} for testing.")
