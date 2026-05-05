"""
Centralized configuration for the InPost City Analyzer.

All constants, default parameters, color palettes, and file paths
are defined here to avoid magic numbers scattered across the codebase.
"""

from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "src" / "data" / "inpost_parcel_locker.csv"

# ---------------------------------------------------------------------------
# InPost API
# ---------------------------------------------------------------------------
API_URL = "https://api-global-points.easypack24.net/v1/points"
DEFAULT_CITY = "Kraków"
DEFAULT_POINT_TYPE = "parcel_locker"
DEFAULT_PER_PAGE = 25

# ---------------------------------------------------------------------------
# Fleet parameters (defaults — can be overridden by dashboard sliders)
# ---------------------------------------------------------------------------
TOTAL_CARS: int = 50
EV_COUNT: int = 25
HYBRID_COUNT: int = 10

# ---------------------------------------------------------------------------
# KMeans clustering
# ---------------------------------------------------------------------------
DEFAULT_N_ZONES: int = 5
MIN_ZONES: int = 3
MAX_ZONES: int = 10

# ---------------------------------------------------------------------------
# Air quality
# ---------------------------------------------------------------------------
SMOG_SCORE_MAP: Dict[str, int] = {
    "VERY_GOOD": 1,
    "GOOD": 2,
    "VERY_BAD": 3,
}

AIR_QUALITY_COLORS: Dict[str, List[int]] = {
    "VERY_GOOD": [46, 204, 113, 200],   # Green  — clean air
    "GOOD":      [241, 196, 15, 200],    # Yellow — moderate
    "VERY_BAD":  [231, 76, 60, 200],     # Red    — polluted
}
DEFAULT_AIR_COLOR: List[int] = [149, 165, 166, 200]  # Gray — no data

# ---------------------------------------------------------------------------
# Fleet profile colors (RGBA for pydeck polygons)
# ---------------------------------------------------------------------------
FLEET_PROFILE_COLORS: Dict[str, List[int]] = {
    "EV Dominant":     [46, 204, 113, 60],   # Green translucent
    "Hybrid Mixed":    [52, 152, 219, 60],   # Blue translucent
    "Diesel Dominant": [149, 165, 166, 60],  # Gray translucent
}

# ---------------------------------------------------------------------------
# Dashboard columns displayed in the data table
# ---------------------------------------------------------------------------
DISPLAY_COLUMNS: List[str] = [
    "name",
    "location_type",
    "physical_type_mapped",
    "is_doubled",
    "air_index_level",
    "delivery_zone_id",
    "fleet_profile",
    "recommended_low_interest_box_machines_list",
]

# ---------------------------------------------------------------------------
# Map defaults (Kraków center)
# ---------------------------------------------------------------------------
DEFAULT_LATITUDE: float = 50.06
DEFAULT_LONGITUDE: float = 19.94
DEFAULT_ZOOM: float = 11.5
DEFAULT_PITCH: int = 45

# ---------------------------------------------------------------------------
# Columns selected from the raw API response
# ---------------------------------------------------------------------------
RAW_SELECTED_COLUMNS: List[str] = [
    "name",
    "location_latitude",
    "location_longitude",
    "physical_type_mapped",
    "apm_doubled",
    "recommended_low_interest_box_machines_list",
    "location_type",
    "easy_access_zone",
    "air_index_level",
]
