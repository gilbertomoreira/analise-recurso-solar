"""Extraterrestrial radiation on a horizontal surface (D&B Sec. 1.4 and 1.10).

Energy results are returned in kWh/m2 (daily: kWh/m2/day); 1 kWh = 3.6 MJ.
"""

from __future__ import annotations

import numpy as np

from solarapp.config import KWH_TO_MJ, SOLAR_CONSTANT_W_M2
from solarapp.physics.geometry import (
    MEAN_DAYS,
    ArrayLike,
    cos_zenith,
    declination,
    sunset_hour_angle,
)


def eccentricity_factor(n: ArrayLike) -> ArrayLike:
    """Orbit eccentricity correction E0 = Gon / Gsc, D&B Eq. 1.4.1b.

    E0 = 1 + 0.033 cos(360 n / 365)
    """
    return 1.0 + 0.033 * np.cos(np.radians(360.0 * np.asarray(n, dtype=float) / 365.0))


def normal_extraterrestrial_irradiance(n: ArrayLike, gsc: float = SOLAR_CONSTANT_W_M2) -> ArrayLike:
    """Extraterrestrial normal irradiance Gon [W/m2], D&B Eq. 1.4.1: Gon = Gsc E0."""
    return gsc * eccentricity_factor(n)


def g0_horizontal(
    lat: ArrayLike, n: ArrayLike, omega: ArrayLike, gsc: float = SOLAR_CONSTANT_W_M2
) -> ArrayLike:
    """Instantaneous extraterrestrial irradiance on a horizontal plane G0 [W/m2], Eq. 1.10.1.

    G0 = Gon cos(theta_z), set to 0 when the sun is below the horizon.
    """
    cz = cos_zenith(lat, declination(n), omega)
    return normal_extraterrestrial_irradiance(n, gsc) * np.maximum(cz, 0.0)


def h0_hourly_kwh(
    lat: float, n: float, omega1: float, omega2: float, gsc: float = SOLAR_CONSTANT_W_M2
) -> float:
    """Extraterrestrial irradiation I0 [kWh/m2] between hour angles omega1 < omega2, Eq. 1.10.4.

    I0 = (12*3600/pi) Gsc E0 [cos(phi) cos(delta) (sin w2 - sin w1)
                              + pi (w2 - w1)/180 sin(phi) sin(delta)]   [J/m2]
    The interval is clipped to the daylight period.
    """
    delta = float(declination(n))
    ws = float(sunset_hour_angle(lat, delta))
    w1, w2 = max(omega1, -ws), min(omega2, ws)
    if w2 <= w1:
        return 0.0
    phi, d = np.radians(lat), np.radians(delta)
    bracket = np.cos(phi) * np.cos(d) * (np.sin(np.radians(w2)) - np.sin(np.radians(w1))) + (
        np.pi * (w2 - w1) / 180.0
    ) * np.sin(phi) * np.sin(d)
    joules = 12.0 * 3600.0 / np.pi * gsc * float(eccentricity_factor(n)) * bracket
    return float(joules / 3.6e6)


def h0_daily_kwh(lat: ArrayLike, n: ArrayLike, gsc: float = SOLAR_CONSTANT_W_M2) -> ArrayLike:
    """Daily extraterrestrial irradiation H0 [kWh/m2/day], D&B Eq. 1.10.3.

    H0 = (24*3600/pi) Gsc E0 [cos(phi) cos(delta) sin(ws) + pi ws/180 sin(phi) sin(delta)]
    """
    delta = declination(n)
    ws = sunset_hour_angle(lat, delta)
    phi, d, wsr = np.radians(lat), np.radians(delta), np.radians(ws)
    bracket = np.cos(phi) * np.cos(d) * np.sin(wsr) + wsr * np.sin(phi) * np.sin(d)
    joules = 24.0 * 3600.0 / np.pi * gsc * eccentricity_factor(n) * bracket
    return joules / 3.6e6


def h0_daily_mj(lat: ArrayLike, n: ArrayLike, gsc: float = SOLAR_CONSTANT_W_M2) -> ArrayLike:
    """Daily extraterrestrial irradiation H0 [MJ/m2/day]."""
    return h0_daily_kwh(lat, n, gsc) * KWH_TO_MJ


def h0_monthly_mean_kwh(lat: float, gsc: float = SOLAR_CONSTANT_W_M2) -> np.ndarray:
    """Monthly mean daily H0 [kWh/m2/day] evaluated on the mean day of each month (Table 1.6.1)."""
    return np.asarray(h0_daily_kwh(lat, MEAN_DAYS, gsc))