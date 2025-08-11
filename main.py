# main.py  —— fast startup + lazy model loading + GCS fetch

import os, logging, threading
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Query, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.responses import HTMLResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# === Config via env ===
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "").strip()
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "").strip().strip("/")
# Cloud Run filesystem: only /tmp is writable; default there
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/tmp/models")).resolve()

# expected filenames (kept same as before)
EXPECTED_FILES = {
    "hgb": "hgb_model.joblib",
    "rf":  "rf_model.joblib",
    "seg": "segmented_model.joblib",
}

# Resolved paths (may update after GCS fetch)
def _model_paths(root: Path):
    return {k: root / v for k, v in EXPECTED_FILES.items()}

MODEL_FILES = _model_paths(BASE_DIR)  # prefer baked-in files if present
MODELS = {}
_LOAD_LOCK = threading.Lock()
_FETCH_LOCK = threading.Lock()
_FETCHED = False


def fetch_models_from_gcs():
    """Download expected model files from GCS into MODEL_DIR if missing."""
    global _FETCHED, MODEL_FILES
    if not MODEL_BUCKET or not MODEL_PREFIX:
        logging.info("GCS model download skipped (MODEL_BUCKET/MODEL_PREFIX not set).")
        return

    with _FETCH_LOCK:
        if _FETCHED:
            return

        # If all files already exist somewhere (baked into image), skip
        already_here = all((_model_paths(MODEL_DIR)[k].exists() or MODEL_FILES[k].exists())
                           for k in EXPECTED_FILES)
        if already_here:
            logging.info("All model files already present; skipping GCS fetch.")
            _FETCHED = True
            return

        try:
            from google.cloud import storage  # local import; add to requirements.txt
        except Exception as e:
            logging.exception("google-cloud-storage not available: %s", e)
            return

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        client = storage.Client()
        bucket = client.bucket(MODEL_BUCKET)

        wanted = set(EXPECTED_FILES.values())
        found_any = False

        # List once; download only the files we expect (by basename)
        prefix = f"{MODEL_PREFIX}/" if MODEL_PREFIX else ""
        logging.info("Fetching models from gs://%s/%s", MODEL_BUCKET, prefix)
        for blob in client.list_blobs(MODEL_BUCKET, prefix=prefix):
            name = os.path.basename(blob.name)
            if name in wanted:
                dest = MODEL_DIR / name
                try:
                    blob.download_to_filename(str(dest))
                    logging.info("Downloaded %s -> %s", blob.name, dest)
                    found_any = True
                except Exception as e:
                    logging.exception("Failed to download %s: %s", blob.name, e)

        # Update MODEL_FILES to point at MODEL_DIR first (so loads from downloaded set)
        MODEL_FILES = _model_paths(MODEL_DIR)

        # Log any missing models
        for k, p in MODEL_FILES.items():
            if not p.exists():
                logging.error("Missing model after GCS fetch: %s at %s", k, p)

        _FETCHED = True
        if not found_any:
            logging.error("No expected model files found at gs://%s/%s", MODEL_BUCKET, prefix)


def load_models():
    # Ensure we attempted a GCS fetch first (if configured)
    if not _FETCHED:
        fetch_models_from_gcs()

    with _LOAD_LOCK:
        if MODELS:
            return
        import joblib  # heavy import local

        # If nothing in MODEL_DIR, fall back to BASE_DIR (baked-in)
        candidates = list(MODEL_FILES.items())
        # Also check baked-in paths to be safe
        baked_files = _model_paths(BASE_DIR)
        for k, baked in baked_files.items():
            if k not in dict(candidates) or not dict(candidates)[k].exists():
                candidates.append((k, baked))

        # Deduplicate by key, prefer first occurrence (downloaded beats baked)
        seen = set()
        ordered = []
        for k, p in candidates:
            if k not in seen:
                seen.add(k)
                ordered.append((k, p))

        for name, path in ordered:
            if not path.exists():
                logging.error("Missing model file for %s at: %s", name, path)
        for name, path in ordered:
            if not path.exists():
                continue
            try:
                MODELS[name] = joblib.load(path)
                logging.info("Loaded %s from %s", name, path)
            except Exception as e:
                logging.exception("Failed to load %s: %s", name, e)

def ensure_loaded():
    if not MODELS:
        load_models()

@app.on_event("startup")
def startup():
    # Kick off background download quickly to reduce p50 on first request
    threading.Thread(target=fetch_models_from_gcs, daemon=True).start()
    if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
        threading.Thread(target=load_models, daemon=True).start()

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

@app.get("/")
def root():
    return filter_form()

# ---------- GeoJSON filtering & map (unchanged) ----------
def filter_geojson_on_demand(file_id: str, borough: str, year: int):
    import json
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
    if os.path.exists(map_file):
        try:
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

# ---------- Prediction API ----------
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
    ensure_loaded()

    import pandas as pd  # local import
    import numpy as np

    df = pd.DataFrame([req.dict()])

    m = MODELS.get(model)
    if m is None:
        raise HTTPException(status_code=400, detail=f"unknown model '{model}'")

    yhat = m.predict(df)[0]
    try:
        y = float(np.expm1(yhat))  # if trained on log1p
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
