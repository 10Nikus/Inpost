"""
InPost City Analyzer — Streamlit Dashboard

A dispatcher-control dashboard for managing parcel locker delivery
zones in Kraków. Visualizes K-Means clustering with fleet allocation
(EV / Hybrid / Diesel) driven by air quality data.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
from typing import Tuple, List

from src.config import (
    DEFAULT_N_ZONES,
    DISPLAY_COLUMNS,
    EV_COUNT,
    HYBRID_COUNT,
    MAX_ZONES,
    MIN_ZONES,
    RAW_CSV_PATH,
    TOTAL_CARS,
)
from src.data.transformer import add_display_columns, fill_missing_air_data, fill_missing_data
from src.models.clustering import get_zone_centers, run_clustering_pipeline
from src.visualization.charts import (
    air_quality_pie_chart,
    fleet_allocation_bar_chart,
    smog_vs_fleet_scatter,
    zone_machine_count_chart,
)
from src.visualization.map_layers import (
    build_deck,
    build_view_state,
    create_scatterplot_layer,
    create_zone_label_layer,
    create_zone_polygon_layer,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InPost City Analyzer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Header accent bar ---- */
    .stApp > header {
        background: linear-gradient(90deg, #FFCD00 0%, #FF8C00 100%);
    }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255, 205, 0, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #a0a0b0 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #FFCD00 !important;
        font-weight: 700;
        font-size: 1.8rem !important;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255, 205, 0, 0.1);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #FFCD00;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(26, 26, 46, 0.5);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFCD00 0%, #FF8C00 100%) !important;
        color: #1a1a2e !important;
        font-weight: 700;
    }

    /* ---- Expander ---- */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #FFCD00;
    }

    /* ---- Hide Streamlit branding ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data() -> pd.DataFrame:
    """Load and pre-process the raw CSV (cached)."""
    df = pd.read_csv(RAW_CSV_PATH)
    df = fill_missing_data(df)
    df = fill_missing_air_data(df)
    return df


raw_df = load_raw_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🎛️ Dispatcher Panel")
st.sidebar.markdown("Manage Green-SLA metrics and fleet optimization in real time.")

# Filters
st.sidebar.header("🔍 Filters")
show_congested = st.sidebar.checkbox("⚠️ Show congested only (is_doubled = 1)", value=False)
air_options = sorted(raw_df["air_index_level"].dropna().unique().tolist())
air_filter = st.sidebar.multiselect("🌬️ Air Quality Level", options=air_options, default=air_options)
location_filter = st.sidebar.radio(
    "🏢 Location Type",
    options=["All", "Outdoor (Flexible)", "Indoor (Strict SLA)"],
)
show_zones = st.sidebar.checkbox("🗺️ Show delivery zone polygons", value=False)

# Fleet configuration (visible only when zones are toggled on)
if show_zones:
    st.sidebar.header("⚡ Fleet Configuration")
    n_zones = st.sidebar.slider("Number of Zones (K)", MIN_ZONES, MAX_ZONES, DEFAULT_N_ZONES)
    total_cars = st.sidebar.slider("Total Vehicles", 10, 100, TOTAL_CARS)
    ev_slider = st.sidebar.slider("Electric Vehicles (EV)", 0, total_cars, min(EV_COUNT, total_cars))
    hybrid_slider = st.sidebar.slider(
        "Hybrid Vehicles", 0, total_cars - ev_slider, min(HYBRID_COUNT, total_cars - ev_slider)
    )
else:
    # Use defaults when sliders are hidden
    n_zones = DEFAULT_N_ZONES
    total_cars = TOTAL_CARS
    ev_slider = min(EV_COUNT, total_cars)
    hybrid_slider = min(HYBRID_COUNT, total_cars - ev_slider)

# ── Clustering pipeline (re-runs when sliders change) ────────────────────────
@st.cache_data
def run_pipeline(
    _raw_df: pd.DataFrame,
    n_zones: int,
    total_cars: int,
    ev_count: int,
    hybrid_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict], pd.DataFrame]:
    """Wrapper for clustering pipeline with Streamlit caching."""
    enriched_df, zone_stats, model, polygons = run_clustering_pipeline(
        _raw_df,
        n_zones=n_zones,
        total_cars=total_cars,
        ev_count=ev_count,
        hybrid_count=hybrid_count,
    )
    enriched_df = add_display_columns(enriched_df)
    centers = get_zone_centers(model)
    # Add labels to centers
    profile_map = dict(zip(zone_stats["delivery_zone_id"], zone_stats["fleet_profile"]))
    centers["label"] = centers["delivery_zone_id"].map(
        lambda zid: f"Zone {zid}\n{profile_map.get(zid, '')}"
    )
    return enriched_df, zone_stats, polygons, centers


enriched_df, zone_stats, polygons, centers = run_pipeline(
    raw_df, n_zones, total_cars, ev_slider, hybrid_slider
)

# ── Apply filters ────────────────────────────────────────────────────────────
filtered_df = enriched_df.copy()

if show_congested:
    filtered_df = filtered_df[filtered_df["is_doubled"] == 1]

if air_filter:
    filtered_df = filtered_df[filtered_df["air_index_level"].isin(air_filter)]

if location_filter == "Outdoor":
    filtered_df = filtered_df[filtered_df["location_type"] == "Outdoor"]
elif location_filter == "Indoor":
    filtered_df = filtered_df[filtered_df["location_type"] == "Indoor"]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🗺️ InPost City Analyzer — Kraków")
st.markdown(
    "Real-time parcel locker network capacity and CO₂-optimized fleet management dashboard."
)

# KPI row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Visible Lockers", f"{len(filtered_df):,}")
col2.metric("High Pollution", len(filtered_df[filtered_df["air_index_level"] == "VERY_BAD"]))
col3.metric("Indoor", len(filtered_df[filtered_df["location_type"] == "Indoor"]))
col4.metric("Congested Points", len(filtered_df[filtered_df["is_doubled"] == 1]))

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_map, tab_analytics, tab_data = st.tabs(["🗺️ Map", "📊 Analytics", "📋 Data"])

# ── Map Tab ──────────────────────────────────────────────────────────────────
with tab_map:
    layers = [create_scatterplot_layer(filtered_df)]

    if show_zones and polygons:
        layers.append(create_zone_polygon_layer(polygons))
        layers.append(create_zone_label_layer(centers))

    view = build_view_state(filtered_df)
    deck = build_deck(layers, view)
    st.pydeck_chart(deck, width="stretch")

    with st.expander("ℹ️ Map Legend"):
        legend_cols = st.columns(3)
        legend_cols[0].markdown("🟢 **Green dot** — Very Good air quality")
        legend_cols[1].markdown("🟡 **Yellow dot** — Good air quality")
        legend_cols[2].markdown("🔴 **Red dot** — Very Bad air quality")
        st.markdown("---")
        zone_cols = st.columns(3)
        zone_cols[0].markdown("🟩 **Green zone** — EV Dominant fleet")
        zone_cols[1].markdown("🟦 **Blue zone** — Hybrid Mixed fleet")
        zone_cols[2].markdown("⬜ **Gray zone** — Diesel Dominant fleet")

# ── Analytics Tab ────────────────────────────────────────────────────────────
with tab_analytics:
    if filtered_df.empty:
        st.warning("⚠️ No lockers match your filter criteria. Please adjust the filters to view analytics.")
    else:
        st.subheader("Fleet Allocation Overview")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(
                fleet_allocation_bar_chart(zone_stats),
                width="stretch",
            )
        with chart_col2:
            st.plotly_chart(
                air_quality_pie_chart(filtered_df),
                width="stretch",
            )

        st.markdown("---")

        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            st.plotly_chart(
                zone_machine_count_chart(zone_stats),
                width="stretch",
            )
        with chart_col4:
            st.plotly_chart(
                smog_vs_fleet_scatter(zone_stats),
                width="stretch",
            )

        # Zone summary table
        st.subheader("Zone Statistics")
        st.dataframe(
            zone_stats,
            width="stretch",
            hide_index=True,
            column_config={
                "delivery_zone_id": st.column_config.TextColumn("Zone ID"),
                "avg_smog": st.column_config.NumberColumn("Avg Smog", format="%.2f"),
                "machine_count": st.column_config.NumberColumn("Machines"),
            }
        )

# ── Data Tab ─────────────────────────────────────────────────────────────────
with tab_data:
    st.subheader(f"Filtered Data — {len(filtered_df):,} lockers")

    available_cols = [c for c in DISPLAY_COLUMNS if c in filtered_df.columns]
    st.dataframe(
        filtered_df[available_cols],
        width="stretch",
        hide_index=True,
        height=500,
        column_config={
            "delivery_zone_id": st.column_config.TextColumn("Zone ID"),
            "is_doubled": st.column_config.NumberColumn("Is Doubled", format="%d"),
        }
    )

    csv_bytes = filtered_df[available_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download filtered CSV",
        data=csv_bytes,
        file_name="inpost_filtered_export.csv",
        mime="text/csv",
    )