# main.py  —— fast startup + lazy model loading

import os, logging, threading
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Query, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.responses import HTMLResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# Defer heavy loads: models will be loaded on-demand or in a background thread
MODEL_FILES = {
    "hgb": BASE_DIR / "hgb_model.joblib",
    "rf":  BASE_DIR / "rf_model.joblib",
    "seg": BASE_DIR / "segmented_model.joblib",
}
MODELS = {}
_LOAD_LOCK = threading.Lock()

def load_models():
    """Load all models once. Safe to call multiple times."""
    with _LOAD_LOCK:
        if MODELS:
            return
        import joblib  # heavy import kept local
        for name, path in MODEL_FILES.items():
            try:
                MODELS[name] = joblib.load(path)
                logging.info("Loaded %s from %s", name, path.name)
            except Exception as e:
                logging.exception("Failed to load %s: %s", name, e)

def ensure_loaded():
    """Lazy load models when first needed."""
    if not MODELS:
        load_models()

@app.on_event("startup")
def startup():
    # Service should become healthy immediately. Models can warm up in background.
    if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
        threading.Thread(target=load_models, daemon=True).start()

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    # Simple health endpoint for Cloud Run checks
    return "ok"

@app.get("/")
def root():
    return filter_form()

def filter_geojson_on_demand(file_id: str, borough: str, year: int):
    """Download GeoJSON on-demand and filter it by borough and year."""
    import json  # local import
    import gdown
    logging.info("Filtering on demand: borough=%s, year=%s", borough, year)

    temp_path = "/tmp/traffic_temp.geojson"
    gdown.download(id=file_id, output=temp_path, quiet=True)

    filtered = []
    with open(temp_path, "r", encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except Exception as e:
            logging.exception("Error loading GeoJSON: %s", e)
            return {"type": "FeatureCollection", "features": []}

        for feature in raw.get("features", []):
            props = feature.get("properties", {})
            b = props.get("Borough", "").lower()
            ts = props.get("Timestamp", "")
            try:
                dt = datetime.fromisoformat(ts)
                if b == borough.lower() and dt.year == year:
                    filtered.append(feature)
            except ValueError:
                continue

    try:
        os.remove(temp_path)
    except OSError:
        pass

    return {"type": "FeatureCollection", "features": filtered}

@app.get("/map")
def get_map(borough: str = Query(), year: int = Query(), background_tasks: BackgroundTasks = None):
    logging.info("/map hit: borough=%s, year=%s", borough, year)

    display_data = filter_geojson_on_demand(
        file_id="1wO3NjqVdg_GUpoEv1JpJHxZoV20Zz-Uq",
        borough=borough,
        year=year,
    )
    if not display_data["features"]:
        return JSONResponse(status_code=404, content={"error": f"No features found for {borough} in {year}"})

    # Import folium only when needed
    import folium
    m = folium.Map(location=[40.739, -73.952], zoom_start=12)

    def style_function(feature):
        volume = feature["properties"].get("Volume", 0)
        if volume > 20:
            color = "red"
        elif volume > 10:
            color = "orange"
        elif volume > 5:
            color = "yellow"
        else:
            color = "green"
        return {"color": color, "weight": 5, "opacity": 0.8}

    folium.GeoJson(
        display_data,
        name="Traffic Data",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["Street", "From", "To", "Volume", "Timestamp", "Direction", "Borough"],
            aliases=["Street:", "From:", "To:", "Volume:", "Timestamp:", "Direction:", "Borough:"],
            localize=True,
        ),
    ).add_to(m)

    map_file = "/tmp/traffic_map.html"
    try:
        if os.path.exists(map_file):
            os.remove(map_file)
    except PermissionError:
        return JSONResponse(content={"error": "Map file in use. Try again."}, status_code=500)

    m.save(map_file)
    if background_tasks:
        background_tasks.add_task(os.remove, map_file)
    return FileResponse(map_file, background=background_tasks)

@app.get("/filter", response_class=HTMLResponse)
def filter_form():
    boroughs = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    years = list(range(2014, 2024))
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

# ==== Prediction API ====

class PredictRequest(BaseModel):
    hour_sin: float
    hour_cos: float
    wd_sin: float
    wd_cos: float
    month_sin: float
    month_cos: float
    vol_lag_1: float
    vol_roll_3h: float
    vol_roll_24h: float

class PredictResponse(BaseModel):
    volume: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model: str = "rf"):
    # Lazy load models only when prediction is called
    ensure_loaded()

    # Import heavy libs only here
    import pandas as pd
    import numpy as np

    df = pd.DataFrame([req.dict()])

    m = MODELS.get(model)
    if m is None:
        raise HTTPException(status_code=400, detail=f"unknown model '{model}'")

    yhat = m.predict(df)[0]
    # If model was trained on log1p(volume), return expm1 back to raw scale.
    try:
        y = float(np.expm1(yhat))
    except Exception:
        y = float(yhat)
    return PredictResponse(volume=y)

@app.get("/ping")
def ping():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
