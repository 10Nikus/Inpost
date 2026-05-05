"""
Unit tests for ``src.data.fetcher``.

API calls are mocked — no real network traffic during tests.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data.fetcher import fetch_inpost_points, to_dataframe


def _make_api_page(items: list, page: int, total_pages: int) -> dict:
    """Helper: build a mock API response page."""
    return {
        "items": items,
        "meta": {"page": page, "total_pages": total_pages},
    }


class TestFetchInpostPoints:
    """Tests for ``fetch_inpost_points``."""

    @patch("src.data.fetcher.requests.get")
    def test_single_page(self, mock_get: MagicMock):
        item = {"name": "KRA01A", "location": {"latitude": 50.0, "longitude": 19.9}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_api_page([item], 1, 1)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_inpost_points(per_page=10)

        assert len(result["items"]) == 1
        assert result["items"][0]["name"] == "KRA01A"
        mock_get.assert_called_once()

    @patch("src.data.fetcher.requests.get")
    def test_pagination(self, mock_get: MagicMock):
        items_p1 = [{"name": "KRA01A"}]
        items_p2 = [{"name": "KRA02B"}]

        resp_p1 = MagicMock()
        resp_p1.json.return_value = _make_api_page(items_p1, 1, 2)
        resp_p1.raise_for_status = MagicMock()

        resp_p2 = MagicMock()
        resp_p2.json.return_value = _make_api_page(items_p2, 2, 2)
        resp_p2.raise_for_status = MagicMock()

        mock_get.side_effect = [resp_p1, resp_p2]

        result = fetch_inpost_points(per_page=1)

        assert len(result["items"]) == 2
        assert mock_get.call_count == 2

    @patch("src.data.fetcher.requests.get")
    def test_api_error_raises(self, mock_get: MagicMock):
        mock_get.side_effect = requests.RequestException("Server error")

        with pytest.raises(requests.RequestException, match="Server error"):
            fetch_inpost_points()

    @patch("src.data.fetcher.requests.get")
    def test_empty_response(self, mock_get: MagicMock):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_api_page([], 1, 1)
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_inpost_points()
        assert len(result["items"]) == 0


class TestToDataframe:
    """Tests for ``to_dataframe``."""

    def test_flattens_nested_json(self):
        data = {
            "items": [
                {
                    "name": "KRA01A",
                    "location": {"latitude": 50.0, "longitude": 19.9},
                }
            ]
        }
        df = to_dataframe(data)
        assert "location_latitude" in df.columns
        assert "location_longitude" in df.columns
        assert len(df) == 1

    def test_empty_items(self):
        df = to_dataframe({"items": []})
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    def test_missing_items_key(self):
        df = to_dataframe({})
        assert len(df) == 0
