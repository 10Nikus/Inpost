# InPost City Analyzer — Kraków

## Author

- **Name:** Nikodem Kijas
- **Email:** kijasnikodem@gmail.com

## Overview

A data-driven dispatcher dashboard for managing InPost parcel locker delivery zones in Kraków. It solves the problem of fleet allocation by using **K-Means clustering** to group lockers into delivery zones, and then applying a **greedy fleet allocation algorithm** (EV / Hybrid / Diesel) that is optimized by real-time air quality data (prioritizing EVs for highly polluted areas).

## Demo & Description

### What it does and how it works
The project features an interactive Streamlit dashboard designed for logistics dispatchers. It allows real-time management of parcel locker network capacity and CO₂-optimized fleet assignment.

**Features:**
- **Real-time map visualization** — Interactive pydeck map with parcel locker dots colored by air quality.
- **K-Means delivery zones** — Geographic clustering with convex hull polygon overlays.
- **Fleet optimization** — Greedy algorithm allocates EVs to the most polluted zones first.
- **Interactive sliders** — Adjust the number of zones (K), total vehicles, and EV/Hybrid counts live.
- **Analytics dashboard** — Plotly charts for fleet allocation, air quality distribution, and pollution vs demand.

### Dashboard Tabs
- **🗺️ Map**: Interactive map with locker dots (colored by air quality) and zone polygons (colored by fleet profile).
- **📊 Analytics**: Stacked bar charts for fleet allocation, donut charts for air quality, and summary tables.
- **📋 Data**: Filterable data table with CSV download capability.

### Data Pipeline Architecture
```text
InPost API → fetch_inpost_points() → to_dataframe()
    → select_columns() → fill_missing_data() → fill_missing_air_data() (Haversine nearest-neighbor)
    → fit_kmeans_zones() → allocate_fleet() → get_zone_polygons()
    → Streamlit Dashboard
```

### Project Structure
```text
inpost/
├── src/                          # Core business logic package
│   ├── config.py                 # Centralized constants and parameters
│   ├── data/
│   │   ├── fetcher.py            # InPost API client with pagination
│   │   └── transformer.py        # Data cleaning & feature engineering
│   ├── models/
│   │   └── clustering.py         # K-Means clustering & fleet allocation
│   └── visualization/
│       ├── map_layers.py         # Pydeck layer builders
│       └── charts.py             # Plotly chart builders
├── tests/                        # pytest test suite
├── app.py                        # Streamlit dashboard entry point
└── requirements.txt              # Python dependencies
```

🚀 **[Live Demo on Streamlit Community Cloud](https://inpost-jwka7qjzze5voxkgw62luv.streamlit.app/)

## Technologies

| Layer | Technology | Reason |
|-------|------------|--------|
| **Dashboard** | Streamlit, Pydeck, Plotly | Streamlit allows rapid prototyping of interactive data apps. Pydeck and Plotly provide excellent and performant mapping and charting tools out of the box. |
| **ML & Math** | scikit-learn (K-Means), SciPy (ConvexHull) | Standard, highly optimized libraries for clustering and generating geographical bounds for zones. |
| **Data processing** | Pandas, NumPy | Best-in-class tools for tabular data manipulation and vectorized operations (like the Haversine formula calculation). |
| **API** | Requests | Simple and robust HTTP client for fetching data from the ShipX API. |
| **Testing** | pytest | Powerful and easy-to-use testing framework for ensuring pipeline reliability. |

## How to run

### Prerequisites

- Python 3.10+
- `pip` package manager

### Build & run

1. **Clone the repository and enter the directory:**
   ```bash
   git clone <your-repo-url>
   cd inpost
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Fetch fresh data from the InPost API:**
   ```bash
   python -m src.data.fetcher
   ```
   *(Note: The repository includes pre-fetched `inpost_parcel_locker.csv` if you wish to skip this step).*

4. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```
   The app will automatically open in your browser at `http://localhost:8501`.

### Configuration
You can adjust default fleet parameters and clustering settings in two ways:
- Live via the **Dashboard sliders** in the sidebar.
- Persistently in `src/config.py`.

### Running Tests
The project includes a comprehensive test suite (100% passing) with mocked API calls and data transform tests. To run them:
```bash
pytest tests/ -v
```

## What I would do with more time

If I had another week, I would focus on:
- **Last-Mile Routing Optimization:** Implementing a routing algorithm (e.g., using OSRM or the Google Maps API) to calculate precise driving routes for individual couriers. This routing would be applied directly on top of the established delivery zones, ensuring that electric vehicles (EVs) are optimally routed specifically through the areas with the highest pollution levels where their zero-emission capabilities are needed most.

## AI usage

I used an AI coding assistant (Claude / Gemini via an IDE integration) to assist with:
- **Dashboard Prototyping:** Generating the initial layout and boilerplate for the Streamlit dashboard, which I later manually refined and styled to fit the project's specific needs.
- **Map Polygons:** Assistance with the mathematical calculations required to generate convex hull polygons representing the delivery zones on the map.
- **Code Review:** Reviewing my architecture and checking for best practices before submission.
- **Refactoring:** Reformatting this `README.md` file to strictly adhere to the provided internship template structure, ensuring all my existing documentation was mapped correctly to the required sections.
- **Typing & Linting:** Minor enhancements to type hinting in a few models to improve static analysis.
