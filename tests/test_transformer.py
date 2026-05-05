"""
Unit tests for ``src.data.transformer``.
"""

import pandas as pd
import pytest

from src.config import AIR_QUALITY_COLORS, DEFAULT_AIR_COLOR, RAW_SELECTED_COLUMNS
from src.data.transformer import (
    add_display_columns,
    fill_missing_air_data,
    fill_missing_data,
    run_transform_pipeline,
    select_columns,
)


class TestSelectColumns:
    """Tests for ``select_columns``."""

    def test_keeps_expected_columns(self, sample_raw_df: pd.DataFrame):
        result = select_columns(sample_raw_df)
        assert list(result.columns) == RAW_SELECTED_COLUMNS

    def test_does_not_mutate_input(self, sample_raw_df: pd.DataFrame):
        original_cols = list(sample_raw_df.columns)
        select_columns(sample_raw_df)
        assert list(sample_raw_df.columns) == original_cols


class TestFillMissingData:
    """Tests for ``fill_missing_data``."""

    def test_creates_is_doubled(self, sample_raw_df: pd.DataFrame):
        result = fill_missing_data(sample_raw_df)
        assert "is_doubled" in result.columns

    def test_drops_apm_doubled(self, sample_raw_df: pd.DataFrame):
        result = fill_missing_data(sample_raw_df)
        assert "apm_doubled" not in result.columns

    def test_is_doubled_values(self, sample_raw_df: pd.DataFrame):
        result = fill_missing_data(sample_raw_df)
        # KRA01A and KRA03C have apm_doubled set → is_doubled = 1
        assert result.loc[0, "is_doubled"] == 1
        assert result.loc[1, "is_doubled"] == 0
        assert result.loc[2, "is_doubled"] == 1

    def test_physical_type_no_nan(self, sample_raw_df: pd.DataFrame):
        result = fill_missing_data(sample_raw_df)
        assert result["physical_type_mapped"].isna().sum() == 0

    def test_does_not_mutate_input(self, sample_raw_df: pd.DataFrame):
        original = sample_raw_df.copy()
        fill_missing_data(sample_raw_df)
        pd.testing.assert_frame_equal(sample_raw_df, original)


class TestFillMissingAirData:
    """Tests for ``fill_missing_air_data``."""

    def test_no_nulls_after_fill(self, sample_raw_df: pd.DataFrame):
        df = fill_missing_data(sample_raw_df)
        result = fill_missing_air_data(df)
        assert result["air_index_level"].isna().sum() == 0

    def test_preserves_existing_values(self, sample_raw_df: pd.DataFrame):
        df = fill_missing_data(sample_raw_df)
        original_known = df[df["air_index_level"].notna()]["air_index_level"].tolist()
        result = fill_missing_air_data(df)
        result_known = result.loc[
            df["air_index_level"].notna().values, "air_index_level"
        ].tolist()
        assert result_known == original_known

    def test_all_have_air_returns_unchanged(self):
        """When no values are missing, the function should be a no-op."""
        df = pd.DataFrame(
            {
                "location_latitude": [50.0, 50.1],
                "location_longitude": [19.0, 19.1],
                "air_index_level": ["GOOD", "VERY_GOOD"],
            }
        )
        result = fill_missing_air_data(df)
        assert result["air_index_level"].tolist() == ["GOOD", "VERY_GOOD"]


class TestAddDisplayColumns:
    """Tests for ``add_display_columns``."""

    def test_color_mapping_known_levels(self, sample_transformed_df: pd.DataFrame):
        df = fill_missing_air_data(sample_transformed_df)
        result = add_display_columns(df)
        # First row is VERY_GOOD → green color
        assert result.loc[0, "color"] == AIR_QUALITY_COLORS["VERY_GOOD"]

    def test_color_mapping_unknown_level(self):
        df = pd.DataFrame(
            {
                "air_index_level": ["UNKNOWN"],
                "physical_type_mapped": [4.0],
                "recommended_low_interest_box_machines_list": [None],
            }
        )
        result = add_display_columns(df)
        assert result.loc[0, "color"] == DEFAULT_AIR_COLOR

    def test_radius_calculated(self, sample_transformed_df: pd.DataFrame):
        result = add_display_columns(sample_transformed_df)
        assert "radius" in result.columns
        # physical_type_mapped 3.0 → radius 45
        assert result.loc[0, "radius"] == 45.0

    def test_recommendation_fallback(self, sample_transformed_df: pd.DataFrame):
        result = add_display_columns(sample_transformed_df)
        # Row 1 had NaN recommendation
        assert result.loc[1, "recommended_low_interest_box_machines_list"] != ""


class TestTransformPipeline:
    """Integration test for ``run_transform_pipeline``."""

    def test_full_pipeline_no_errors(self, sample_raw_df: pd.DataFrame):
        result = run_transform_pipeline(sample_raw_df)
        assert len(result) == len(sample_raw_df)
        assert "is_doubled" in result.columns
        assert "color" in result.columns
        assert "radius" in result.columns
        assert result["air_index_level"].isna().sum() == 0
