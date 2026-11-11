"""Place-name geocoding via Open-Meteo (no API key)."""

from __future__ import annotations

from dataclasses import dataclass

import requests

from solarapp.config import GEOCODING_URL, HTTP_TIMEOUT_S
from solarapp.data.nasa_power import build_session


@dataclass(frozen=True)
class Place:
    name: str
    admin1: str
    country: str
    latitude: float  # [deg], north positive
    longitude: float  # [deg], east positive
    elevation: float | None  # [m]

    @property
    def label(self) -> str:
        parts = [p for p in (self.name, self.admin1, self.country) if p]
        return ", ".join(parts)


class GeocodingError(RuntimeError):
    """Raised when geocoding fails."""


def search_places(
    query: str,
    count: int = 10,
    session: requests.Session | None = None,
    country_code: str | None = None,
) -> list[Place]:
    """Return up to `count` places matching `query` (Portuguese names).

    `country_code` is an ISO-3166 alpha-2 code (e.g. "BR") to restrict results to one country.
    Uses the shared session (system trust store, retries) unless one is given.
    """
    query = query.strip()
    if len(query) < 2:
        return []
    http = session or build_session()
    params = {"name": query, "count": count, "language": "pt", "format": "json"}
    if country_code:
        params["countryCode"] = country_code
    try:
        resp = http.get(
            GEOCODING_URL,
            params=params,
            timeout=HTTP_TIMEOUT_S,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except (requests.RequestException, ValueError) as exc:
        raise GeocodingError(f"Falha na geocodificação: {exc}") from exc
    return [
        Place(
            name=r.get("name", ""),
            admin1=r.get("admin1", "") or "",
            country=r.get("country", "") or "",
            latitude=float(r["latitude"]),
            longitude=float(r["longitude"]),
            elevation=r.get("elevation"),
        )
        for r in results
    ]
