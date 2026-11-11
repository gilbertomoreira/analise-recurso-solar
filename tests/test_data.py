"""NASA POWER client and cache tests (HTTP mocked)."""

from datetime import date

import numpy as np
import pytest
import requests
import responses

from solarapp.data.cache import JsonCache
from solarapp.data.geocoding import GeocodingError, search_places
from solarapp.data.nasa_power import NasaPowerClient, NasaPowerError

MONTHLY_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
HOURLY_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def _monthly_payload() -> dict:
    ghi = {f"2020{m:02d}": 5.0 + m * 0.1 for m in range(1, 13)}
    ghi["202013"] = 5.65
    ghi["202003"] = -999.0
    kt = {k: 0.5 for k in ghi}
    return {"properties": {"parameter": {"ALLSKY_SFC_SW_DWN": ghi, "ALLSKY_KT": kt}}}


@pytest.fixture
def client(tmp_path):
    return NasaPowerClient(cache=JsonCache(tmp_path), timeout=5)


@responses.activate
def test_monthly_parsing_fill_values_and_annual(client):
    responses.add(responses.GET, MONTHLY_URL, json=_monthly_payload())
    monthly, annual = client.fetch_monthly(-7.12, -34.88, 2020, 2020)
    assert len(monthly) == 12
    assert np.isnan(monthly.loc[(2020, 3), "ghi"])  # -999 -> NaN
    assert monthly.loc[(2020, 1), "ghi"] == pytest.approx(5.1)
    assert annual.loc[2020, "ghi"] == pytest.approx(5.65)
    q = responses.calls[0].request.url
    assert "community=RE" in q and "ALLSKY_SFC_SW_DIFF" in q


@responses.activate
def test_cache_avoids_second_call(client):
    responses.add(responses.GET, MONTHLY_URL, json=_monthly_payload())
    client.fetch_monthly(-7.12, -34.88, 2020, 2020)
    client.fetch_monthly(-7.12, -34.88, 2020, 2020)
    assert len(responses.calls) == 1


@responses.activate
def test_daily_index_and_columns(client):
    payload = {
        "properties": {
            "parameter": {"ALLSKY_SFC_SW_DWN": {"20200101": 4.6, "20200102": -999.0}}
        }
    }
    responses.add(responses.GET, DAILY_URL, json=payload)
    df = client.fetch_daily(-7.12, -34.88, date(2020, 1, 1), date(2020, 1, 2))
    assert df.index[0] == np.datetime64("2020-01-01")
    assert df["ghi"].iloc[0] == 4.6 and np.isnan(df["ghi"].iloc[1])


@responses.activate
def test_hourly_index(client):
    payload = {"properties": {"parameter": {"ALLSKY_SFC_SW_DWN": {"2020010112": 800.0}}}}
    responses.add(responses.GET, HOURLY_URL, json=payload)
    df = client.fetch_hourly(-7.12, -34.88, date(2020, 1, 1), date(2020, 1, 1))
    assert df.index[0].hour == 12 and df["ghi"].iloc[0] == 800.0


def test_hourly_span_limit(client):
    with pytest.raises(ValueError):
        client.fetch_hourly(0, 0, date(2020, 1, 1), date(2022, 1, 1))


@pytest.mark.parametrize("lat,lon", [(91, 0), (0, 181)])
def test_invalid_coordinates(client, lat, lon):
    with pytest.raises(ValueError):
        client.fetch_monthly(lat, lon, 2020, 2020)


@responses.activate
def test_http_error_raises(client):
    responses.add(responses.GET, MONTHLY_URL, status=422, body="bad request")
    with pytest.raises(NasaPowerError, match="422"):
        client.fetch_monthly(0, 0, 2020, 2020)


@responses.activate
def test_connection_error_raises(client):
    responses.add(responses.GET, MONTHLY_URL, body=requests.ConnectionError("down"))
    with pytest.raises(NasaPowerError, match="conexão"):
        client.fetch_monthly(0, 0, 2020, 2020)


@responses.activate
def test_malformed_json_raises(client):
    responses.add(responses.GET, MONTHLY_URL, json={"oops": 1})
    with pytest.raises(NasaPowerError, match="inesperado"):
        client.fetch_monthly(0, 0, 2020, 2020)


def test_cache_expiry(tmp_path):
    cache = JsonCache(tmp_path, ttl_days=0)
    cache.set({"a": 1}, {"x": 1})
    assert cache.get({"a": 1}) is None
    assert JsonCache(tmp_path).get({"a": 1}) == {"x": 1}
    assert JsonCache(tmp_path).clear() == 1


@responses.activate
def test_geocoding():
    responses.add(
        responses.GET,
        "https://geocoding-api.open-meteo.com/v1/search",
        json={
            "results": [
                {
                    "name": "João Pessoa",
                    "admin1": "Paraíba",
                    "country": "Brasil",
                    "latitude": -7.115,
                    "longitude": -34.861,
                }
            ]
        },
    )
    places = search_places("João Pessoa")
    assert places[0].label == "João Pessoa, Paraíba, Brasil"
    assert places[0].latitude == pytest.approx(-7.115)


@responses.activate
def test_geocoding_no_results_and_error():
    responses.add(responses.GET, "https://geocoding-api.open-meteo.com/v1/search", json={})
    assert search_places("xyzxyz") == []
    responses.replace(
        responses.GET, "https://geocoding-api.open-meteo.com/v1/search", status=500
    )
    with pytest.raises(GeocodingError):
        search_places("abc")


def test_tls_adapter_does_not_load_certifi_file():
    """Regression: certifi's cacert.pem hangs when the venv is on a cloud drive (Google Drive)."""
    from solarapp.data.nasa_power import _SystemTrustAdapter

    class Conn:
        ca_certs = "should-be-cleared"
        ca_cert_dir = "x"
        cert_reqs = None

    conn = Conn()
    _SystemTrustAdapter().cert_verify(conn, "https://power.larc.nasa.gov", True, None)
    assert conn.ca_certs is None and conn.ca_cert_dir is None
    assert conn.cert_reqs == "CERT_REQUIRED"