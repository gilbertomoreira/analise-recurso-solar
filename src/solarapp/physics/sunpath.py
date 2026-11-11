"""Sun-path (solar chart) data for a location. D&B Sec. 1.6 (Eqs. 1.6.5-1.6.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarapp.physics.geometry import (
    MEAN_DAYS,
    declination,
    solar_altitude,
    solar_azimuth,
    sunset_hour_angle,
)

# Solstices and equinoxes (day numbers) + mean days used for the chart.
KEY_DAYS = {"Solstício de junho (21/jun)": 172, "Equinócio (21/mar)": 80,
            "Solstício de dezembro (21/dez)": 355}


def day_path(lat: float, n: int, step_min: float = 10.0) -> pd.DataFrame:
    """Sun position along day n (sun above horizon only).

    Columns: solar_time_h, altitude_deg, azimuth_deg (0 = south, east negative, west positive),
    azimuth_north_deg (0 = north, clockwise, 0..360).
    """
    delta = float(declination(n))
    ws = float(sunset_hour_angle(lat, delta))
    t = np.arange(12.0 - ws / 15.0, 12.0 + ws / 15.0 + 1e-9, step_min / 60.0)
    omega = 15.0 * (t - 12.0)
    alt = np.asarray(solar_altitude(lat, delta, omega))
    az = np.asarray(solar_azimuth(lat, delta, omega))
    keep = alt >= 0
    return pd.DataFrame(
        {
            "solar_time_h": t[keep],
            "altitude_deg": alt[keep],
            "azimuth_deg": az[keep],
            "azimuth_north_deg": (az[keep] + 180.0) % 360.0,
        }
    )


def sun_path_table(lat: float, days: dict[str, int] | None = None) -> pd.DataFrame:
    """Concatenate day_path for several days with a 'day_label' column."""
    days = days or KEY_DAYS
    frames = [day_path(lat, n).assign(day_label=label, n=n) for label, n in days.items()]
    return pd.concat(frames, ignore_index=True)


def hourly_lines(lat: float, hours: range = range(6, 19)) -> pd.DataFrame:
    """Analemma-like lines of constant solar hour across the year (mean days of each month)."""
    rows = []
    for h in hours:
        omega = 15.0 * (h - 12)
        for month, n in enumerate(MEAN_DAYS, start=1):
            delta = float(declination(n))
            alt = float(solar_altitude(lat, delta, omega))
            if alt >= 0:
                az = float(solar_azimuth(lat, delta, omega))
                rows.append((h, month, alt, az, (az + 180.0) % 360.0))
    return pd.DataFrame(
        rows, columns=["solar_hour", "month", "altitude_deg", "azimuth_deg", "azimuth_north_deg"]
    )