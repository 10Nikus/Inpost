"""
Fetch InPost parcel locker data from the public ShipX API.

This module handles pagination, error handling, and JSON → DataFrame
conversion. It is designed to be called as a standalone script or
imported by other modules.
"""

import logging
from typing import Optional

import pandas as pd
import requests

from src.config import API_URL, DEFAULT_CITY, DEFAULT_PER_PAGE, DEFAULT_POINT_TYPE, RAW_CSV_PATH

logger = logging.getLogger(__name__)


def fetch_inpost_points(
    city: str = DEFAULT_CITY,
    point_type: str = DEFAULT_POINT_TYPE,
    per_page: int = DEFAULT_PER_PAGE,
    timeout: int = 30,
) -> dict:
    """Fetch all InPost points by paginating through the ShipX API.

    Parameters
    ----------
    city : str
        City name to filter by (default: Kraków).
    point_type : str
        Point type filter (default: parcel_locker).
    per_page : int
        Number of results per page.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    dict
        Full API response with *all* items aggregated across pages.

    Raises
    ------
    requests.RequestException
        If any API request fails.
    """
    all_items: list = []
    page = 1

    while True:
        params = {
            "per_page": per_page,
            "page": page,
            "city": city,
            "type": point_type,
        }

        try:
            response = requests.get(API_URL, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("API request failed on page %d: %s", page, exc)
            raise

        data = response.json()
        items = data.get("items", [])
        all_items.extend(items)

        meta = data.get("meta", {})
        total_pages = meta.get("total_pages", 1)
        logger.info(
            "Page %d/%d — fetched %d points (%d total)",
            page,
            total_pages,
            len(items),
            len(all_items),
        )

        if page >= total_pages:
            break
        page += 1

    data["items"] = all_items
    return data


def to_dataframe(data: dict) -> pd.DataFrame:
    """Convert the API JSON response into a flat pandas DataFrame.

    Nested JSON keys are separated by underscores (e.g. ``location_latitude``).
    """
    return pd.json_normalize(data.get("items", []), sep="_")


def run_fetch_pipeline(
    city: str = DEFAULT_CITY,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience wrapper: fetch → flatten → optionally save.

    Parameters
    ----------
    city : str
        City to fetch data for.
    output_path : str, optional
        If provided, save the raw DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Raw, un-transformed DataFrame.
    """
    data = fetch_inpost_points(city=city)
    df = to_dataframe(data)

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info("Saved raw data to %s", output_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_fetch_pipeline(output_path=RAW_CSV_PATH)
