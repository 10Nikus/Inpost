import requests
import numpy as np
import pandas as pd


API_URL = "https://api-shipx-pl.easypack24.net/v1/points"


def fetch_inpost_points(
    city: str = "Kraków",
    point_type: str = "parcel_locker",
    per_page: int = 25
) -> dict:
    """Fetch all InPost points by paginating through the API."""
    all_items = []
    page = 1

    while True:
        params = {
            "per_page": per_page,
            "page": page,
            "city": city,
            "type": point_type
        }

        try:
            response = requests.get(API_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"API request failed on page {page}: {e}")
            raise

        data = response.json()
        items = data.get("items", [])
        all_items.extend(items)

        meta = data.get("meta", {})
        total_pages = meta.get("total_pages", 1)
        print(f"Page {page}/{total_pages} — fetched {len(items)} points ({len(all_items)} total)")

        if page >= total_pages:
            break
        page += 1

    data["items"] = all_items
    return data


def to_dataframe(data: dict) -> pd.DataFrame:
    """Convert API JSON response to a pandas DataFrame."""
    return pd.json_normalize(data.get("items", []), sep="_")


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the DataFrame to keep only selected columns."""
    return df[['name',
        'location_latitude',
        'location_longitude',
        'physical_type_mapped',
        'apm_doubled',
        'recommended_low_interest_box_machines_list',
        'location_type',
        'easy_access_zone',
        'air_index_level']].copy()


def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values for apm_doubled and physical_type_mapped."""
    df["is_doubled"] = df["apm_doubled"].notna().astype(int)
    df.drop(columns=["apm_doubled"], inplace=True)
    df["physical_type_mapped"] = df["physical_type_mapped"].fillna(df["physical_type_mapped"].mode()[0])
    return df


def fill_missing_air_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing air_index_level with the value from the nearest locker that has a sensor."""
    has_air = df[df["air_index_level"].notna()]
    missing_air = df[df["air_index_level"].isna()]

    if has_air.empty or missing_air.empty:
        return df

    # Convert degrees to radians for haversine
    lat1 = np.radians(missing_air["location_latitude"].values[:, np.newaxis])
    lon1 = np.radians(missing_air["location_longitude"].values[:, np.newaxis])
    lat2 = np.radians(has_air["location_latitude"].values[np.newaxis, :])
    lon2 = np.radians(has_air["location_longitude"].values[np.newaxis, :])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distances = 2 * np.arcsin(np.sqrt(a))

    # Find index of nearest locker with sensor for each missing row
    nearest_idx = distances.argmin(axis=1)
    df.loc[missing_air.index, "air_index_level"] = has_air["air_index_level"].values[nearest_idx]

    return df


if __name__ == "__main__":
    data = fetch_inpost_points()
    df = to_dataframe(data)
    df_transformed = transform_data(df)
    df_filled = fill_missing_data(df_transformed)
    df_filled = fill_missing_air_data(df_filled)
    df_filled.to_csv("inpost_parcel_locker.csv", index=False)
