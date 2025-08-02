from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from fastapi import Query
from fastapi.responses import FileResponse
import folium
import os
from datetime import datetime
import gdown

from starlette.responses import HTMLResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Backend starting...")

print("Downloading GeoJSON from Google Drive...")

@app.get("/")
def root():
    return filter_form()

'''
Filters GeoJSON based on the borough and year to avoid memory issues upon deployment.
Args:
    file_id(str): Google Drive file ID
    borough(str): Borough to filter on
    year(int): Year to filter on
Returns:
    dict: Filtered GeoJSON'''
def filter_geojson_on_demand(file_id: str, borough: str, year: int):
    print(f"Filtering on demand: borough={borough}, year={year}")
    temp_path = "traffic_temp.geojson"

    # Download to temp file
    gdown.download(id=file_id, output=temp_path, quiet=True)

    filtered_features = []
    with open(temp_path, "r", encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except Exception as e:
            print("Error loading GeoJSON:", e)
            return {"type": "FeatureCollection", "features": []}

        for feature in raw.get("features", []):
            props = feature.get("properties", {})
            b = props.get("Borough", "").lower()
            ts = props.get("Timestamp", "")
            try:
                dt = datetime.fromisoformat(ts)
                if b == borough.lower() and dt.year == year:
                    filtered_features.append(feature)
            except ValueError:
                continue

    os.remove(temp_path)
    return {
        "type": "FeatureCollection",
        "features": filtered_features
    }


'''
An endpoint that generates a visual representation of traffic data given the borough
and year
Args:
    borough(str)
    year(int)
'''
@app.get("/map")
def get_map(borough: str = Query(), year: int = Query()):
    print(f"/map endpoint hit: borough={borough}, year={year}")

    file_id = "1wO3NjqVdg_GUpoEv1JpJHxZoV20Zz-Uq"
    display_data = filter_geojson_on_demand(file_id, borough, year)

    if not display_data["features"]:
        return JSONResponse(
            status_code=404,
            content={"error": f"No features found for {borough} in {year}"}
        )

    print(f"Filtered down to {len(display_data['features'])} features.")

    # Generate map
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

    map_file = "/tmp/traffic_map.html"
    if os.path.exists(map_file):
        try:
            os.remove(map_file)
        except PermissionError:
            return JSONResponse(content={"error": "Map file in use. Try again."}, status_code=500)

    m.save(map_file)
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

# Health check to make sure server is running
@app.get("/ping")
def ping():
    print("Ping received", flush=True)
    return {"status": "ok"}

