from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import folium
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Backend starting...")

geojson_path = os.path.join(os.getcwd(), "data", "processed", "traffic_data_osm_lines_clean.geojson")

print(f"Loading GeoJSON from: {geojson_path}")
with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)
print(f"Loaded {len(geojson_data.get('features', []))} features from GeoJSON")

@app.get("/map")
def get_map():
    print("/map endpoint hit: Generating map...")

    m = folium.Map(location=[40.739, -73.952], zoom_start=12)

    def style_function(feature):
        volume = feature['properties'].get('Volume', 0)
        if volume > 20:
            color = 'red'
        elif volume > 10:
            color = 'orange'
        elif volume > 5:
            color = 'yellow'
        else:
            color = 'green'
        return {
            'color': color,
            'weight': 5,
            'opacity': 0.8
        }

    folium.GeoJson(
        geojson_data,
        name="Traffic Data",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["Street", "From", "To", "Volume", "Timestamp", "Direction", "Borough"],
            aliases=["Street:", "From:", "To:", "Volume:", "Timestamp:", "Direction:", "Borough:"],
            localize=True
        )
    ).add_to(m)

    map_file = "traffic_map.html"
    if os.path.exists(map_file):
        try:
            os.remove(map_file)
            print("Previous map file removed.")
        except PermissionError:
            print("File in use. Close it and try again.")
            return {"error": "File in use. Close it and retry."}

    m.save(map_file)
    print("Map generated and saved as traffic_map.html")
    return FileResponse(map_file)

@app.get("/geojson")
def get_geojson():
    print("/geojson endpoint hit: Returning GeoJSON data")
    return JSONResponse(content=geojson_data)
