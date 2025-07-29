# AI4All Traffic & Weather Analysis

**Data-driven analysis of NYC traffic and weather patterns using machine learning, data engineering, and geospatial visualization.**

---

##  Project Overview

This project analyzes the correlation between **NYC traffic congestion** and **weather conditions** using:
- Historical traffic and weather data
- Predictive machine learning models
- Geospatial and time-series feature engineering
- Web-based map visualizations

---

## 📁 Directory Structure

```text
AI4AllProj/
├── backend/           # Core backend logic: ML models, merging, feature pipelines
├── data/              # Raw and processed data files
│   ├── raw/
│   ├── processed/
│   └── cache/
├── experimental/       # Jupyter notebooks, training experiments
├── frontend/          # Simple web-based visualization interface
├── scripts/           # Data processing, transformation, and utilities
│   └── test/          # Test scripts
├── .env               # Environment variables
├── .gitignore
├── README.md
└── requirements.txt
```

---

##  Key Components

###  Backend
Located in the `backend/` directory:
- `app.py`: Main FastAPI app (if used)
- `linear_regression.py` & `random_forest.py`: Model implementations
- `features.py`: Feature extraction & transformations
- `weather_merge.py` & `raw_merge.py`: Data merge utilities

###  Experiments
Located in `experimental/`:
- `train_model.py`: Model training script
- `models.py`: Model loading and saving logic
- `correlation_analysis.ipynb`: Notebook with EDA and correlation matrices

###  Scripts
Located in `scripts/`:
- `convert_csv_to_geojson.py`: Converts CSV to GeoJSON
- `feature_engineering.py`: Adds temporal and weather-based features
- `enrich_weather.py`: Merges external weather APIs
- `point_to_linestring.py`, `downsize_data.py`: Spatial/size optimizations

###  Frontend
Located in `frontend/`:
- `index.html`: Landing page
- `traffic_map.html`: Interactive Folium map

---

##  Data Sources

- **Traffic Data:**  
  NYC DOT: [Automated Traffic Volume Counts](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/btm5-4yqh)

- **Weather Data:**  
  Open-Meteo API & historical CSVs  
  Merged across 2014–2024 across multiple VMs for resilience

---

## Technologies

| Type          | Tools/Libraries                                 |
|---------------|-------------------------------------------------|
| Language      | Python 3.13, HTML, JavaScript                   |
| ML Libraries  | `scikit-learn`, `pandas`, `numpy`, `matplotlib` |
| Geo/Mapping   | `folium`, `geopandas`, `osmnx`, `shapely`       |
| API / Backend | `FastAPI`, `requests`, `uvicorn`                |
| DevTools      | Git, VSCode, GitHub, Jupyter Notebooks          |

---

##  Contributors

- **Justin Forbes**  
- **Zahir Humphries**
- **Kate Liu**
- **Nnaemeka Okonkwo**
- **Sania Shree**

---

## License

