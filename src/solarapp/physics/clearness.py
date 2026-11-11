"""Clearness index (D&B Sec. 2.9)."""

from __future__ import annotations

import numpy as np

from solarapp.physics.extraterrestrial import h0_daily_kwh, h0_monthly_mean_kwh
from solarapp.physics.geometry import MEAN_DAYS, ArrayLike  # noqa: F401


def clearness_index(h: ArrayLike, h0: ArrayLike) -> ArrayLike:
    """Clearness index Kt = H / H0 [-], D&B Eq. 2.9.1 (daily) and 2.9.2 (monthly mean).

    H and H0 must share units (e.g. kWh/m2/day). NaN where H0 <= 0 (polar night).
    """
    h, h0 = np.asarray(h, dtype=float), np.asarray(h0, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(h0 > 0, h / h0, np.nan)


def daily_kt(h_kwh: ArrayLike, lat: float, n: ArrayLike) -> ArrayLike:
    """Daily Kt from GHI [kWh/m2/day] and day number(s) n."""
    return clearness_index(h_kwh, h0_daily_kwh(lat, n))


def monthly_kt(h_monthly_kwh: ArrayLike, lat: float) -> ArrayLike:
    """Monthly mean Kt from 12 monthly mean daily GHI [kWh/m2/day] (H0 on the mean day)."""
    return clearness_index(h_monthly_kwh, h0_monthly_mean_kwh(lat))