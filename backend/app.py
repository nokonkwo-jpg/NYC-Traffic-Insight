from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from fastapi import Query
from fastapi.responses import FileResponse
import folium
import os
from datetime import datetime

from starlette.responses import HTMLResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Backend starting...")

geojson_path = os.path.join(os.getcwd(), "data", "processed", "output", "traffic_data_osm_lines_clean.geojson")

print(f"Loading GeoJSON from: {geojson_path}")
with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)
print(f"Loaded {len(geojson_data.get('features', []))} features from GeoJSON")

'''
An endpoint that generates a visual representation of traffic data given the borough
and year
Args:
    borough(str)
    year(int)
'''
@app.get("/map")
def get_map(borough: str = Query(), year: int = Query()):
    print(f"/map endpoint hit: Generating map... (borough={borough}, year={year})")

    # Filter only if parameters are provided
    if borough and year:
        filtered_features = []
        for feature in geojson_data["features"]:
            props = feature.get("properties", {})
            b = props.get("Borough", "").lower()
            timestamp = props.get("Timestamp", "")

            try:
                dt = datetime.fromisoformat(timestamp)
                if b == borough.lower() and dt.year == year:
                    filtered_features.append(feature)
            except ValueError:
                continue

        if not filtered_features:
            return JSONResponse(
                status_code=404,
                content={"error": f"No features found for {borough} in {year}"}
            )

        display_data = {
            "type": "FeatureCollection",
            "features": filtered_features
        }
        print(f"Filtered down to {len(filtered_features)} features.")
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Borough and year query parameters are required."}
        )

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
        display_data,
        name="Traffic Data",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["Street", "From", "To", "Volume", "Timestamp", "Direction", "Borough"],
            aliases=["Street:", "From:", "To:", "Volume:", "Timestamp:", "Direction:", "Borough:"],
            localize=True
        )
    ).add_to(m)

    map_file = "../frontend/traffic_map.html"
    if os.path.exists(map_file):
        try:
            os.remove(map_file)
            print("Previous map file removed.")
        except PermissionError:
            print("File in use. Close it and try again.")
            return JSONResponse(content={"error": "File in use. Close it and retry."}, status_code=500)

    m.save(map_file)
    print(f"Map saved as {map_file}")
    return FileResponse(map_file)

'''
A form users fill out to query the /map endpoint
'''
@app.get("/filter", response_class=HTMLResponse)
def filter_form():
    boroughs = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    years = list(range(2014, 2024))  # or use logic to generate based on your data

    options_html = lambda items: "\n".join(f'<option value="{item}">{item}</option>' for item in items)

    return f"""
    <html>
    <head><title>Filter Map</title></head>
    <body>
        <h2>Select Borough and Year</h2>
        <form action="/map" method="get">
            <label for="borough">Borough:</label>
            <select name="borough" required>
                {options_html(boroughs)}
            </select><br><br>

            <label for="year">Year:</label>
            <select name="year" required>
                {options_html(years)}
            </select><br><br>

            <button type="submit">Generate Map</button>
        </form>
    </body>
    </html>
    """

'''
Returns JSON of all GeoJSON used
'''
@app.get("/geojson")
def get_geojson():
    print("/geojson endpoint hit: Returning GeoJSON data")
    return JSONResponse(content=geojson_data)

# Health check to make sure server is running
@app.get("/ping")
def ping():
    print("Ping received", flush=True)
    return {"status": "ok"}

