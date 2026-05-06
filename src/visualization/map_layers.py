"""
Pydeck map layer builders for the Streamlit dashboard.

Each function returns a configured ``pdk.Layer`` instance that can be
combined in a ``pdk.Deck``.
"""

from typing import List, Optional

import pandas as pd
import pydeck as pdk

from src.config import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, DEFAULT_PITCH, DEFAULT_ZOOM


def create_scatterplot_layer(df: pd.DataFrame) -> pdk.Layer:
    """Individual parcel locker dots, sized by physical type and colored by air quality."""
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[location_longitude, location_latitude]",
        get_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=1,
        radius_min_pixels=3,
        radius_max_pixels=30,
        line_width_min_pixels=1,
    )


def create_zone_polygon_layer(polygons: List[dict]) -> pdk.Layer:
    """Translucent convex-hull overlay for each delivery zone."""
    return pdk.Layer(
        "PolygonLayer",
        data=polygons,
        get_polygon="polygon",
        get_fill_color="color",
        get_line_color=[255, 255, 255, 80],
        line_width_min_pixels=2,
        pickable=False,
        stroked=True,
        filled=True,
        extruded=False,
        opacity=0.5,
    )


def create_zone_label_layer(centers_df: pd.DataFrame) -> pdk.Layer:
    """Text labels at zone centroids showing fleet profile."""
    return pdk.Layer(
        "TextLayer",
        data=centers_df,
        get_position="[center_lon, center_lat]",
        get_text="label",
        get_size=14,
        get_color=[255, 255, 255, 230],
        get_angle=0,
        get_text_anchor="'middle'",
        get_alignment_baseline="'center'",
        pickable=False,
        billboard=True,
    )


def build_view_state(
    df: Optional[pd.DataFrame] = None,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    zoom: float = DEFAULT_ZOOM,
    pitch: int = DEFAULT_PITCH,
) -> pdk.ViewState:
    """Build a ``ViewState`` centered on the data or on Kraków defaults."""
    if df is not None and not df.empty:
        latitude = float(df["location_latitude"].mean())
        longitude = float(df["location_longitude"].mean())

    return pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=zoom,
        pitch=pitch,
    )


def build_deck(
    layers: List[pdk.Layer],
    view_state: pdk.ViewState,
    tooltip: Optional[dict] = None,
) -> pdk.Deck:
    """Assemble a ``pdk.Deck`` with CARTO dark basemap."""
    default_tooltip = {
        "html": (
            "<b>ID:</b> {name} <br/>"
            "<b>Air Quality:</b> {air_index_level} <br/>"
            "<b>Type:</b> {location_type} (Size: {physical_type_mapped}) <br/>"
            "<b>Congested:</b> {is_doubled} <br/>"
            "<b>Zone:</b> {delivery_zone_id} — {fleet_profile} <br/>"
            "<hr/>"
            "<b>➡️ Redirect recommendation:</b><br/>"
            "{recommended_low_interest_box_machines_list}"
        ),
        "style": {
            "backgroundColor": "#1a1a2e",
            "color": "white",
            "font-family": "'Inter', Helvetica, Arial, sans-serif",
            "border-radius": "8px",
            "padding": "10px",
        },
    }
    return pdk.Deck(
        map_provider="carto",
        map_style="dark",
        initial_view_state=view_state,
        layers=layers,
        tooltip=tooltip or default_tooltip,
    )
