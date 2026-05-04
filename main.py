import requests
import json

import pandas as pd


API_URL = "https://api-shipx-pl.easypack24.net/v1/points"


def fetch_inpost_points(per_page=10, page=1) -> pd.Json:
    """Fetch InPost parcel locker points from the API."""
    params = {
        "per_page": per_page,
        "page": page,
    }

    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    return response.json()

def to_dataframe(data) -> pd.Json:
    """Convert API JSON response to a pandas DataFrame."""
    return pd.json_normalize(data.get("items", []), sep="_")

def transfrom_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the DataFrame to remove unwanted columns and rows."""

    return df


if __name__ == "__main__":
    data = fetch_inpost_points(per_page=50000, page=1)
    df = to_dataframe(data)
    
