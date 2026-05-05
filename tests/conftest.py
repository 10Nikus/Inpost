"""
Shared pytest fixtures for the InPost test suite.

Provides small, representative DataFrames that mirror the real data
structure without requiring API calls or CSV files.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Minimal DataFrame mimicking the raw API output after ``json_normalize``."""
    return pd.DataFrame(
        {
            "name": ["KRA01A", "KRA02B", "KRA03C", "KRA04D", "KRA05E"],
            "location_latitude": [50.085, 50.060, 50.040, 50.095, 50.070],
            "location_longitude": [19.920, 19.940, 19.970, 19.900, 19.960],
            "physical_type_mapped": [3.0, 4.0, None, 6.0, 4.0],
            "apm_doubled": ["KRA99Z", None, "KRA88Y", None, None],
            "recommended_low_interest_box_machines_list": [
                "KRA10,KRA11",
                None,
                "KRA20",
                None,
                None,
            ],
            "location_type": ["Outdoor", "Indoor", "Outdoor", "Outdoor", "Indoor"],
            "easy_access_zone": [True, False, True, True, False],
            "air_index_level": ["VERY_GOOD", "GOOD", "VERY_BAD", None, "GOOD"],
        }
    )


@pytest.fixture
def sample_transformed_df(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame after ``fill_missing_data`` — no ``apm_doubled``, has ``is_doubled``."""
    df = sample_raw_df.copy()
    df["is_doubled"] = df["apm_doubled"].notna().astype(int)
    df.drop(columns=["apm_doubled"], inplace=True)
    df["physical_type_mapped"] = df["physical_type_mapped"].fillna(
        df["physical_type_mapped"].mode().iloc[0]
    )
    return df


@pytest.fixture
def sample_zone_df(sample_transformed_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame after clustering — includes ``delivery_zone_id``."""
    df = sample_transformed_df.copy()
    # Fill missing air_index_level for clustering tests
    df["air_index_level"] = df["air_index_level"].fillna("GOOD")
    df["delivery_zone_id"] = [0, 1, 0, 1, 0]
    return df
