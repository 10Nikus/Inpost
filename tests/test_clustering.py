"""
Unit tests for ``src.models.clustering``.
"""

import pandas as pd
import pytest

from src.models.clustering import (
    allocate_fleet,
    fit_kmeans_zones,
    get_zone_centers,
    get_zone_polygons,
    run_clustering_pipeline,
)


@pytest.fixture
def clustering_input_df() -> pd.DataFrame:
    """DataFrame with enough rows for 2-cluster K-Means."""
    return pd.DataFrame(
        {
            "name": [f"KRA{i:02d}" for i in range(20)],
            "location_latitude": [50.0 + i * 0.01 for i in range(20)],
            "location_longitude": [19.9 + (i % 5) * 0.01 for i in range(20)],
            "air_index_level": (["VERY_GOOD"] * 8 + ["GOOD"] * 7 + ["VERY_BAD"] * 5),
            "location_type": ["Outdoor"] * 10 + ["Indoor"] * 10,
            "physical_type_mapped": [4.0] * 20,
            "is_doubled": [0] * 15 + [1] * 5,
            "recommended_low_interest_box_machines_list": [None] * 20,
        }
    )


class TestFitKmeansZones:
    """Tests for ``fit_kmeans_zones``."""

    def test_adds_zone_column(self, clustering_input_df: pd.DataFrame):
        df, _model = fit_kmeans_zones(clustering_input_df, n_zones=3)
        assert "delivery_zone_id" in df.columns

    def test_correct_zone_count(self, clustering_input_df: pd.DataFrame):
        n = 3
        df, _model = fit_kmeans_zones(clustering_input_df, n_zones=n)
        assert df["delivery_zone_id"].nunique() == n

    def test_all_rows_assigned(self, clustering_input_df: pd.DataFrame):
        df, _model = fit_kmeans_zones(clustering_input_df, n_zones=3)
        assert df["delivery_zone_id"].isna().sum() == 0

    def test_returns_fitted_model(self, clustering_input_df: pd.DataFrame):
        _df, model = fit_kmeans_zones(clustering_input_df, n_zones=3)
        assert model.cluster_centers_.shape == (3, 2)

    def test_does_not_mutate_input(self, clustering_input_df: pd.DataFrame):
        original = clustering_input_df.copy()
        fit_kmeans_zones(clustering_input_df, n_zones=3)
        pd.testing.assert_frame_equal(clustering_input_df, original)


class TestAllocateFleet:
    """Tests for ``allocate_fleet``."""

    def test_total_assigned_within_budget(self, clustering_input_df: pd.DataFrame):
        df, _model = fit_kmeans_zones(clustering_input_df, n_zones=3)
        stats = allocate_fleet(df, total_cars=20, ev_count=10, hybrid_count=5)
        total_assigned = (
            stats["EV_assigned"].sum()
            + stats["Hybrid_assigned"].sum()
            + stats["Diesel_assigned"].sum()
        )
        assert total_assigned <= 20

    def test_ev_priority_for_worst_smog(self, clustering_input_df: pd.DataFrame):
        """The zone with the highest avg_smog should get EVs first."""
        df, _model = fit_kmeans_zones(clustering_input_df, n_zones=2)
        stats = allocate_fleet(df, total_cars=10, ev_count=5, hybrid_count=2)
        worst_zone = stats.sort_values("avg_smog", ascending=False).iloc[0]
        assert worst_zone["EV_assigned"] > 0

    def test_fleet_profile_values(self, clustering_input_df: pd.DataFrame):
        df, _model = fit_kmeans_zones(clustering_input_df, n_zones=3)
        stats = allocate_fleet(df)
        valid_profiles = {"EV Dominant", "Hybrid Mixed", "Diesel Dominant"}
        assert set(stats["fleet_profile"]).issubset(valid_profiles)

    def test_zone_stats_columns(self, clustering_input_df: pd.DataFrame):
        df, _model = fit_kmeans_zones(clustering_input_df, n_zones=3)
        stats = allocate_fleet(df)
        expected_cols = {
            "delivery_zone_id",
            "avg_smog",
            "machine_count",
            "cars_needed",
            "EV_assigned",
            "Hybrid_assigned",
            "Diesel_assigned",
            "fleet_profile",
        }
        assert set(stats.columns) == expected_cols


class TestZoneGeometry:
    """Tests for ``get_zone_centers`` and ``get_zone_polygons``."""

    def test_centers_shape(self, clustering_input_df: pd.DataFrame):
        _df, model = fit_kmeans_zones(clustering_input_df, n_zones=3)
        centers = get_zone_centers(model)
        assert len(centers) == 3
        assert "center_lat" in centers.columns
        assert "center_lon" in centers.columns

    def test_polygons_structure(self, clustering_input_df: pd.DataFrame):
        df, _model = fit_kmeans_zones(clustering_input_df, n_zones=2)
        stats = allocate_fleet(df)
        polygons = get_zone_polygons(df, stats)
        assert isinstance(polygons, list)
        assert len(polygons) > 0
        for poly in polygons:
            assert "polygon" in poly
            assert "color" in poly
            assert "zone_id" in poly
            assert len(poly["polygon"]) >= 4  # at least 3 points + closing point

    def test_polygon_is_closed(self, clustering_input_df: pd.DataFrame):
        df, _model = fit_kmeans_zones(clustering_input_df, n_zones=2)
        stats = allocate_fleet(df)
        polygons = get_zone_polygons(df, stats)
        for poly in polygons:
            assert poly["polygon"][0] == poly["polygon"][-1]


class TestClusteringPipeline:
    """Integration test for ``run_clustering_pipeline``."""

    def test_full_pipeline(self, clustering_input_df: pd.DataFrame):
        enriched, stats, model, polygons = run_clustering_pipeline(
            clustering_input_df, n_zones=2, total_cars=10, ev_count=5, hybrid_count=2
        )
        assert "fleet_profile" in enriched.columns
        assert "delivery_zone_id" in enriched.columns
        assert len(stats) == 2
        assert len(polygons) > 0
        assert model.cluster_centers_.shape == (2, 2)
