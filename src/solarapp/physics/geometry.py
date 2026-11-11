"""Solar geometry. Pure functions; angles in degrees, times in hours.

Sign conventions (Duffie & Beckman): latitude north positive; surface azimuth gamma = 0 facing
the equator... precisely: 0 = south, east negative, west positive (range -180..180); hour angle
negative in the morning; longitude here is geographic (east positive).
Reference: Duffie & Beckman, Solar Engineering of Thermal Processes, 4th ed., Ch. 1.
"""

from __future__ import annotations

from datetime import date

import numpy as np

ArrayLike = float | np.ndarray

DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_CUM_DAYS = np.concatenate(([0], np.cumsum(DAYS_IN_MONTH)[:-1]))
# Recommended average day of each month (D&B Table 1.6.1, Klein 1977).
MEAN_DAYS = np.array([17, 47, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344])


def day_of_year(month: int, day: int) -> int:
    """Day number n of the year (1..365, non-leap), D&B Table 1.6.1."""
    if not (1 <= month <= 12 and 1 <= day <= int(DAYS_IN_MONTH[month - 1])):
        raise ValueError("Data inválida.")
    return int(_CUM_DAYS[month - 1]) + day


def day_of_year_from_date(d: date) -> int:
    """Day number n for a datetime.date; 29 Feb is mapped to 28 Feb (n=59)."""
    if d.month == 2 and d.day == 29:
        return 59
    return day_of_year(d.month, d.day)


def _rad(x: ArrayLike) -> ArrayLike:
    return np.radians(x)


def declination(n: ArrayLike) -> ArrayLike:
    """Solar declination delta [deg], Cooper (1969): D&B Eq. 1.6.1a.

    delta = 23.45 sin(360 (284 + n) / 365)
    """
    return 23.45 * np.sin(_rad(360.0 * (284.0 + np.asarray(n, dtype=float)) / 365.0))


def declination_spencer(n: ArrayLike) -> ArrayLike:
    """Solar declination delta [deg], Spencer (1971) Fourier series: D&B Eq. 1.6.1b."""
    b = _rad((np.asarray(n, dtype=float) - 1.0) * 360.0 / 365.0)
    delta = (
        0.006918
        - 0.399912 * np.cos(b)
        + 0.070257 * np.sin(b)
        - 0.006758 * np.cos(2 * b)
        + 0.000907 * np.sin(2 * b)
        - 0.002697 * np.cos(3 * b)
        + 0.00148 * np.sin(3 * b)
    )
    return np.degrees(delta)


def equation_of_time(n: ArrayLike) -> ArrayLike:
    """Equation of time E [min], D&B Eq. 1.5.3 (Spencer).

    E = 229.2 (0.000075 + 0.001868 cos B - 0.032077 sin B - 0.014615 cos 2B - 0.04089 sin 2B)
    with B = (n - 1) 360 / 365.
    """
    b = _rad((np.asarray(n, dtype=float) - 1.0) * 360.0 / 365.0)
    return 229.2 * (
        0.000075
        + 0.001868 * np.cos(b)
        - 0.032077 * np.sin(b)
        - 0.014615 * np.cos(2 * b)
        - 0.04089 * np.sin(2 * b)
    )


def standard_meridian(utc_offset_h: float) -> float:
    """Longitude of the standard time meridian [deg, east positive] = 15 * UTC offset."""
    return 15.0 * utc_offset_h


def longitude_correction_min(longitude: float, std_meridian: float) -> float:
    """Longitude correction [min] = 4 (L_loc - L_std), both east positive.

    D&B Eq. 1.5.2 uses west-positive longitudes: 4 (L_st - L_loc); same quantity.
    """
    return 4.0 * (longitude - std_meridian)


def solar_time(
    standard_time_h: ArrayLike, n: ArrayLike, longitude: float, std_meridian: float
) -> ArrayLike:
    """Local solar time [h] from standard (clock) time [h], D&B Eq. 1.5.2.

    solar time = standard time + (4 (L_loc - L_std) + E) / 60   (no daylight saving)
    """
    return standard_time_h + (longitude_correction_min(longitude, std_meridian)
                              + equation_of_time(n)) / 60.0


def hour_angle(solar_time_h: ArrayLike) -> ArrayLike:
    """Hour angle omega [deg]: 15 deg per hour from solar noon (negative before noon)."""
    return 15.0 * (np.asarray(solar_time_h, dtype=float) - 12.0)


def sunset_hour_angle(lat: ArrayLike, delta: ArrayLike) -> ArrayLike:
    """Sunset hour angle omega_s [deg], D&B Eq. 1.6.10: cos(ws) = -tan(phi) tan(delta).

    Returns 180 for polar day and 0 for polar night.
    """
    x = -np.tan(_rad(lat)) * np.tan(_rad(delta))
    return np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))


def day_length(lat: ArrayLike, n: ArrayLike) -> ArrayLike:
    """Astronomical day length N [h], D&B Eq. 1.6.11: N = 2 omega_s / 15."""
    return 2.0 * sunset_hour_angle(lat, declination(n)) / 15.0


def cos_zenith(lat: ArrayLike, delta: ArrayLike, omega: ArrayLike) -> ArrayLike:
    """cos(theta_z) = cos(phi) cos(delta) cos(omega) + sin(phi) sin(delta), D&B Eq. 1.6.5."""
    return np.cos(_rad(lat)) * np.cos(_rad(delta)) * np.cos(_rad(omega)) + np.sin(
        _rad(lat)
    ) * np.sin(_rad(delta))


def zenith_angle(lat: ArrayLike, delta: ArrayLike, omega: ArrayLike) -> ArrayLike:
    """Solar zenith angle theta_z [deg], D&B Eq. 1.6.5."""
    return np.degrees(np.arccos(np.clip(cos_zenith(lat, delta, omega), -1.0, 1.0)))


def solar_altitude(lat: ArrayLike, delta: ArrayLike, omega: ArrayLike) -> ArrayLike:
    """Solar altitude alpha_s [deg] = 90 - theta_z (D&B Eq. 1.6.6 text)."""
    return 90.0 - zenith_angle(lat, delta, omega)


def solar_azimuth(lat: ArrayLike, delta: ArrayLike, omega: ArrayLike) -> ArrayLike:
    """Solar azimuth gamma_s [deg], 0 = south, east negative (D&B Eq. 1.6.6).

    gamma_s = sign(omega) |arccos((cos theta_z sin phi - sin delta) / (sin theta_z cos phi))|
    """
    cz = cos_zenith(lat, delta, omega)
    sz = np.sqrt(np.clip(1.0 - cz**2, 0.0, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (cz * np.sin(_rad(lat)) - np.sin(_rad(delta))) / (sz * np.cos(_rad(lat)))
    x = np.where(np.isfinite(x), x, 1.0)
    g = np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))
    return np.where(np.asarray(omega) >= 0, g, -g)