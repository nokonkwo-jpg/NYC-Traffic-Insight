from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import folium
import os, sys
from datetime import datetime
import gdown
import joblib, pandas as pd, numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "traffic_volume_models"))

from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

app = FastAPI()
hgb   = joblib.load("hgb_model.joblib")
rf    = joblib.load("rf_model.joblib")
seg   = joblib.load("segmented_model.joblib")

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
def get_map(borough: str = Query(), year: int = Query(), background_tasks: BackgroundTasks=None):
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
    background_tasks.add_task(os.remove, map_file)
    return FileResponse(map_file, background=background_tasks)


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

# 1) Define the request schema
class PredictRequest(BaseModel):
    hour_sin:    float = Field(..., example=0.0)
    hour_cos:    float = Field(..., example=1.0)
    wd_sin:      float = Field(..., example=0.0)
    wd_cos:      float = Field(..., example=1.0)
    month_sin:   float = Field(..., example=0.5)
    month_cos:   float = Field(..., example=0.866)
    vol_lag_1:   float = Field(..., example=100)
    vol_roll_3h: float = Field(..., example=110)
    vol_roll_24h:float = Field(..., example=115)

# 2) (Optional) Define the response schema
class PredictResponse(BaseModel):
    volume: float = Field(..., example=42.7)

# 3) Wire up the POST endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Turn the incoming JSON into a DataFrame row:
    df = pd.DataFrame([req.dict()])

    # You can choose which model to call here:
    # e.g. use the HGB regressor:
    log_vol = hgb.predict(df)[0]
    raw_vol = float(np.expm1(log_vol))

    # Or to use the segmented model:
    # raw_vol = float(seg_model.predict(df)[0])

    return PredictResponse(volume=raw_vol)

'''
Returns JSON of all GeoJSON used
'''

# Health check to make sure server is running
@app.get("/ping")
def ping():
    print("Ping received", flush=True)
    return {"status": "ok"}

