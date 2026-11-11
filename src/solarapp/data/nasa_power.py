"""NASA POWER client (monthly, daily and hourly point API).

Docs: https://power.larc.nasa.gov/docs/services/api/temporal/ . No API key required.
Missing values are flagged as -999 and converted to NaN.
"""

from __future__ import annotations

import functools
import ssl
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from solarapp.config import (
    HTTP_RETRIES,
    HTTP_TIMEOUT_S,
    MAX_HOURLY_SPAN_DAYS,
    NASA_COMMUNITY,
    NASA_FILL_VALUE,
    NASA_POWER_BASE_URL,
    USER_AGENT,
)
from solarapp.data.cache import JsonCache
from solarapp.data.schema import NASA_TO_INTERNAL

SOLAR_PARAMS = [
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DIFF",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_KT",
    "CLRSKY_SFC_SW_DWN",
    "TOA_SW_DWN",
    "T2M",
    "WS10M",
]
HOURLY_PARAMS = ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DIFF", "ALLSKY_SFC_SW_DNI", "T2M", "WS10M"]


class NasaPowerError(RuntimeError):
    """Raised when the NASA POWER API fails or returns unusable data."""


@functools.lru_cache(maxsize=1)
def _system_ssl_context() -> ssl.SSLContext:
    """Default SSL context (OS trust store), built once per process."""
    return ssl.create_default_context()


class _SystemTrustAdapter(HTTPAdapter):
    """HTTPAdapter that verifies TLS with the operating-system trust store.

    By default `requests` loads certifi's cacert.pem file; when the virtualenv lives on a cloud
    drive (e.g. Google Drive streaming) OpenSSL can hang reading it. The system store avoids that
    and honours corporate/university CA certificates.
    """

    def cert_verify(self, conn, url, verify, cert):
        """Skip the certifi CA file: trust is taken from the shared system SSL context."""
        if verify is True and str(url).lower().startswith("https"):
            conn.cert_reqs = "CERT_REQUIRED"
            conn.ca_certs = None
            conn.ca_cert_dir = None
            return
        super().cert_verify(conn, url, verify, cert)

    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        host_params, pool_kwargs = super().build_connection_pool_key_attributes(
            request, verify, cert
        )
        if verify is True:
            pool_kwargs["ssl_context"] = _system_ssl_context()
        return host_params, pool_kwargs


def build_session() -> requests.Session:
    """HTTP session with retry/backoff on transient errors."""
    retry = Retry(
        total=HTTP_RETRIES,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", _SystemTrustAdapter(max_retries=retry))
    session.headers["User-Agent"] = USER_AGENT
    return session


class NasaPowerClient:
    """Fetches point time series and returns tidy DataFrames with internal column names."""

    def __init__(
        self,
        session: requests.Session | None = None,
        cache: JsonCache | None = None,
        timeout: float = HTTP_TIMEOUT_S,
    ) -> None:
        self.session = session or build_session()
        self.cache = cache or JsonCache()
        self.timeout = timeout

    # ------------------------------------------------------------------ transport
    def _request(self, temporal: str, params: dict[str, Any]) -> dict[str, Any]:
        key = {"temporal": temporal, **params}
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        url = f"{NASA_POWER_BASE_URL}/{temporal}/point"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise NasaPowerError(f"Falha de conexão com a NASA POWER: {exc}") from exc
        if resp.status_code != 200:
            detail = resp.text[:300]
            raise NasaPowerError(f"NASA POWER retornou HTTP {resp.status_code}: {detail}")
        try:
            payload = resp.json()
            payload["properties"]["parameter"]
        except (ValueError, KeyError, TypeError) as exc:
            raise NasaPowerError("Resposta da NASA POWER em formato inesperado.") from exc
        self.cache.set(key, payload)
        return payload

    @staticmethod
    def _base_params(lat: float, lon: float, parameters: list[str]) -> dict[str, Any]:
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("Latitude deve estar em [-90, 90] e longitude em [-180, 180].")
        return {
            "parameters": ",".join(parameters),
            "community": NASA_COMMUNITY,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "format": "JSON",
        }

    @staticmethod
    def _to_frame(payload: dict[str, Any]) -> pd.DataFrame:
        data = payload["properties"]["parameter"]
        df = pd.DataFrame(data)
        df = df.replace(NASA_FILL_VALUE, np.nan)
        return df.rename(columns=NASA_TO_INTERNAL)

    # ------------------------------------------------------------------ public API
    def fetch_monthly(
        self, lat: float, lon: float, start_year: int, end_year: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Monthly means [kWh/m2/day]. Returns (monthly, annual).

        monthly: index (year, month); annual: index year (NASA's "month 13" key).
        """
        if end_year < start_year:
            raise ValueError("Ano final menor que o inicial.")
        params = self._base_params(lat, lon, SOLAR_PARAMS)
        params.update(start=start_year, end=end_year)
        df = self._to_frame(self._request("monthly", params))
        idx = df.index.astype(str)
        df["year"] = idx.str[:4].astype(int)
        df["month"] = idx.str[4:].astype(int)
        annual = df[df["month"] == 13].drop(columns="month").set_index("year")
        monthly = df[df["month"] <= 12].set_index(["year", "month"]).sort_index()
        return monthly, annual

    def fetch_daily(self, lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
        """Daily values [kWh/m2/day], DatetimeIndex."""
        if end < start:
            raise ValueError("Data final anterior à inicial.")
        params = self._base_params(lat, lon, SOLAR_PARAMS)
        params.update(start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"))
        df = self._to_frame(self._request("daily", params))
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
        df.index.name = "date"
        return df.sort_index()

    def fetch_hourly(self, lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
        """Hourly means in local solar time. Irradiance columns in W/m2 (Wh/m2 per hour)."""
        if end < start:
            raise ValueError("Data final anterior à inicial.")
        if (end - start).days + 1 > MAX_HOURLY_SPAN_DAYS:
            raise ValueError(f"Intervalo horário limitado a {MAX_HOURLY_SPAN_DAYS} dias.")
        params = self._base_params(lat, lon, HOURLY_PARAMS)
        params.update(
            start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"), **{"time-standard": "LST"}
        )
        df = self._to_frame(self._request("hourly", params))
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d%H")
        df.index.name = "datetime_lst"
        return df.sort_index()
