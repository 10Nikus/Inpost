"""
K-Means clustering and greedy fleet allocation for delivery zones.

Clusters InPost parcel lockers into geographic zones, then allocates
a mixed fleet (EV / Hybrid / Diesel) using a greedy algorithm that
prioritizes electric vehicles in the most polluted zones.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import KMeans

from src.config import (
    DEFAULT_N_ZONES,
    EV_COUNT,
    FLEET_PROFILE_COLORS,
    HYBRID_COUNT,
    SMOG_SCORE_MAP,
    TOTAL_CARS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def fit_kmeans_zones(
    df: pd.DataFrame,
    n_zones: int = DEFAULT_N_ZONES,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, KMeans]:
    """Fit K-Means on locker coordinates and assign ``delivery_zone_id``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``location_latitude`` and ``location_longitude``.
    n_zones : int
        Number of clusters (K).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, KMeans]
        DataFrame with ``delivery_zone_id`` column and the fitted model.
    """
    df = df.copy()
    coords = df[["location_latitude", "location_longitude"]].values
    model = KMeans(n_clusters=n_zones, random_state=random_state, n_init=10)
    df["delivery_zone_id"] = model.fit_predict(coords)
    return df, model


# ---------------------------------------------------------------------------
# Fleet allocation
# ---------------------------------------------------------------------------

def allocate_fleet(
    df: pd.DataFrame,
    total_cars: int = TOTAL_CARS,
    ev_count: int = EV_COUNT,
    hybrid_count: int = HYBRID_COUNT,
) -> pd.DataFrame:
    """Greedy fleet allocation: assign EV → Hybrid → Diesel by pollution.

    Zones with the worst air quality receive electric vehicles first.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``delivery_zone_id`` and ``air_index_level``.
    total_cars, ev_count, hybrid_count : int
        Fleet composition parameters.

    Returns
    -------
    pd.DataFrame
        Zone-level statistics with fleet allocation columns:
        ``delivery_zone_id``, ``avg_smog``, ``machine_count``,
        ``cars_needed``, ``EV_assigned``, ``Hybrid_assigned``,
        ``Diesel_assigned``, ``fleet_profile``.
    """
    df = df.copy()
    df["air_score"] = df["air_index_level"].map(SMOG_SCORE_MAP)

    zone_stats = (
        df.groupby("delivery_zone_id")
        .agg(avg_smog=("air_score", "mean"), machine_count=("name", "count"))
        .reset_index()
    )

    total_machines = zone_stats["machine_count"].sum()
    zone_stats["cars_needed"] = np.ceil(
        (zone_stats["machine_count"] / total_machines) * total_cars
    ).astype(int)

    # Sort by worst smog first — EVs go there
    zone_stats = zone_stats.sort_values("avg_smog", ascending=False).reset_index(drop=True)

    ice_count = max(total_cars - ev_count - hybrid_count, 0)
    available = {"ev": ev_count, "hybrid": hybrid_count, "ice": ice_count}
    allocations: List[dict] = []

    for _, row in zone_stats.iterrows():
        needed = int(row["cars_needed"])
        alloc = {"ev": 0, "hybrid": 0, "ice": 0}

        for fuel in ("ev", "hybrid", "ice"):
            if needed <= 0:
                break
            take = min(needed, available[fuel])
            alloc[fuel] = take
            available[fuel] -= take
            needed -= take

        if alloc["ev"] >= alloc["hybrid"] and alloc["ev"] >= alloc["ice"]:
            profile = "EV Dominant"
        elif alloc["hybrid"] > alloc["ev"] and alloc["hybrid"] >= alloc["ice"]:
            profile = "Hybrid Mixed"
        else:
            profile = "Diesel Dominant"

        allocations.append(
            {
                "delivery_zone_id": row["delivery_zone_id"],
                "avg_smog": round(row["avg_smog"], 2),
                "machine_count": row["machine_count"],
                "cars_needed": int(row["cars_needed"]),
                "EV_assigned": alloc["ev"],
                "Hybrid_assigned": alloc["hybrid"],
                "Diesel_assigned": alloc["ice"],
                "fleet_profile": profile,
            }
        )

    return pd.DataFrame(allocations)


# ---------------------------------------------------------------------------
# Zone geometry helpers
# ---------------------------------------------------------------------------

def get_zone_centers(model: KMeans) -> pd.DataFrame:
    """Return cluster centroids as a DataFrame.

    Parameters
    ----------
    model : KMeans
        Fitted K-Means model.

    Returns
    -------
    pd.DataFrame
        Columns: ``delivery_zone_id``, ``center_lat``, ``center_lon``.
    """
    centers = model.cluster_centers_
    return pd.DataFrame(
        {
            "delivery_zone_id": range(len(centers)),
            "center_lat": centers[:, 0],
            "center_lon": centers[:, 1],
        }
    )


def get_zone_polygons(
    df: pd.DataFrame,
    zone_stats: pd.DataFrame,
) -> List[dict]:
    """Build convex-hull polygons for each delivery zone.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``delivery_zone_id``, ``location_latitude``,
        ``location_longitude``.
    zone_stats : pd.DataFrame
        Must contain ``delivery_zone_id`` and ``fleet_profile``.

    Returns
    -------
    list[dict]
        Each dict has keys ``polygon`` (list of [lon, lat] pairs) and
        ``color`` (RGBA list) — ready for a pydeck PolygonLayer.
    """
    profile_map = dict(zip(zone_stats["delivery_zone_id"], zone_stats["fleet_profile"]))
    polygons: List[dict] = []

    for zone_id, group in df.groupby("delivery_zone_id"):
        coords = group[["location_longitude", "location_latitude"]].values
        if len(coords) < 3:
            continue

        try:
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices].tolist()
            # Close the polygon
            hull_points.append(hull_points[0])
        except (QhullError, ValueError) as exc:
            logger.warning("Could not compute convex hull for zone %s: %s", zone_id, exc)
            continue

        profile = profile_map.get(zone_id, "Diesel Dominant")
        color = FLEET_PROFILE_COLORS.get(profile, [149, 165, 166, 60])

        polygons.append(
            {
                "polygon": hull_points,
                "color": color,
                "zone_id": zone_id,
                "fleet_profile": profile,
            }
        )

    return polygons


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------

def run_clustering_pipeline(
    df: pd.DataFrame,
    n_zones: int = DEFAULT_N_ZONES,
    total_cars: int = TOTAL_CARS,
    ev_count: int = EV_COUNT,
    hybrid_count: int = HYBRID_COUNT,
) -> Tuple[pd.DataFrame, pd.DataFrame, KMeans, List[dict]]:
    """Run the full clustering + allocation pipeline.

    Returns
    -------
    tuple
        (enriched_df, zone_stats, kmeans_model, zone_polygons)
    """
    df, model = fit_kmeans_zones(df, n_zones=n_zones)
    zone_stats = allocate_fleet(df, total_cars, ev_count, hybrid_count)

    # Merge fleet profile back into locker-level data
    df = df.merge(
        zone_stats[["delivery_zone_id", "fleet_profile"]],
        on="delivery_zone_id",
        how="left",
    )

    polygons = get_zone_polygons(df, zone_stats)
    return df, zone_stats, model, polygons
