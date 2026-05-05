"""
Data transformation and feature engineering for InPost locker data.

Handles column selection, missing-value imputation (including
haversine-based nearest-neighbor air quality fill), and display
column generation for the Streamlit dashboard.
"""

from typing import List

import numpy as np
import pandas as pd

from src.config import (
    AIR_QUALITY_COLORS,
    DEFAULT_AIR_COLOR,
    RAW_SELECTED_COLUMNS,
)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns required for analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from the API.

    Returns
    -------
    pd.DataFrame
        Subset with only ``RAW_SELECTED_COLUMNS``.
    """
    available = [c for c in RAW_SELECTED_COLUMNS if c in df.columns]
    # Also keep 'is_doubled' if it already exists (pre-processed CSV)
    if "is_doubled" in df.columns and "is_doubled" not in available:
        available.append("is_doubled")
    return df[available].copy()


def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create ``is_doubled`` flag and impute ``physical_type_mapped``.

    * ``apm_doubled`` (nullable) → ``is_doubled`` (0 or 1), then drop original.
    * ``physical_type_mapped`` NaN → mode value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``apm_doubled`` and ``physical_type_mapped`` columns.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with ``is_doubled`` added and ``apm_doubled`` removed.
    """
    df = df.copy()

    if "apm_doubled" in df.columns:
        df["is_doubled"] = df["apm_doubled"].notna().astype(int)
        df.drop(columns=["apm_doubled"], inplace=True)
    elif "is_doubled" not in df.columns:
        df["is_doubled"] = 0

    if df["physical_type_mapped"].isna().any():
        mode_val = df["physical_type_mapped"].mode()
        if not mode_val.empty:
            df["physical_type_mapped"] = df["physical_type_mapped"].fillna(mode_val.iloc[0])

    return df


def fill_missing_air_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing ``air_index_level`` using the nearest locker with a sensor.

    Uses the haversine formula to find the geographically closest locker
    that *does* have air quality data and copies its value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``location_latitude``, ``location_longitude``,
        and ``air_index_level`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with no NaN values in ``air_index_level``.
    """
    df = df.copy()
    has_air = df[df["air_index_level"].notna()]
    missing_air = df[df["air_index_level"].isna()]

    if has_air.empty or missing_air.empty:
        return df

    # Haversine distance matrix
    lat1 = np.radians(missing_air["location_latitude"].values[:, np.newaxis])
    lon1 = np.radians(missing_air["location_longitude"].values[:, np.newaxis])
    lat2 = np.radians(has_air["location_latitude"].values[np.newaxis, :])
    lon2 = np.radians(has_air["location_longitude"].values[np.newaxis, :])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distances = 2 * np.arcsin(np.sqrt(a))

    nearest_idx = distances.argmin(axis=1)
    df.loc[missing_air.index, "air_index_level"] = has_air["air_index_level"].values[nearest_idx]

    return df


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns used by the Streamlit dashboard for rendering.

    * ``color`` — RGBA list based on ``air_index_level``.
    * ``radius`` — dot size based on ``physical_type_mapped``.
    * ``recommended_low_interest_box_machines_list`` — friendly default text.

    Parameters
    ----------
    df : pd.DataFrame
        Transformed DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``color`` and ``radius`` columns added.
    """
    df = df.copy()

    def _air_color(level: str) -> List[int]:
        return AIR_QUALITY_COLORS.get(level, DEFAULT_AIR_COLOR)

    df["color"] = df["air_index_level"].apply(_air_color)
    df["radius"] = df["physical_type_mapped"].fillna(4.0) * 15
    df["recommended_low_interest_box_machines_list"] = df[
        "recommended_low_interest_box_machines_list"
    ].fillna("No congestion — no redirect needed")

    return df


def run_transform_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full transformation pipeline.

    ``select_columns → fill_missing_data → fill_missing_air_data → add_display_columns``

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from the API (or CSV).

    Returns
    -------
    pd.DataFrame
        Fully transformed, dashboard-ready DataFrame.
    """
    df = select_columns(df)
    df = fill_missing_data(df)
    df = fill_missing_air_data(df)
    df = add_display_columns(df)
    return df
